#!/usr/bin/env python3
import os
import json
import random
import s3fs
import logging
import traceback
import datetime
import uuid
from io import BytesIO
from urllib.parse import urljoin
from label_studio_sdk.client import LabelStudio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/random_sampler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("random_sampler")

# Constants
MINIO_KEY = 'minioadmin'
MINIO_SECRET = 'minioadmin'
MINIO_ENDPOINT = 'http://minio:9000'
MINIO_PUBLIC_URL = 'http://localhost:9000'
IMAGES_BUCKET = 'production-images'
TRACKING_BUCKET = 'tracking'
RANDOM_SAMPLING_FILE = f'{TRACKING_BUCKET}/random_sampling_tasks.json'
LABEL_STUDIO_URL = "http://label-studio:8080"
API_TOKEN = os.environ.get('LABEL_STUDIO_TOKEN', "ab9927067c51ff279d340d7321e4890dc2841c4a")
SAMPLE_COUNT = int(os.environ.get('SAMPLE_COUNT', '3'))

# Initialize S3 filesystem
try:
    fs = s3fs.S3FileSystem(
        key=MINIO_KEY,
        secret=MINIO_SECRET,
        client_kwargs={
            'endpoint_url': MINIO_ENDPOINT
        }
    )
    logger.info("S3 filesystem initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize S3 filesystem: {e}")
    raise

# Initialize Label Studio client
ls = None
project = None
try:
    ls = LabelStudio(
        base_url=LABEL_STUDIO_URL,
        api_key=API_TOKEN
    )
    logger.info("Label Studio client initialized successfully")
    
    # Find the existing project
    projects = ls.projects.list()
    for p in projects:
        if p.title == "Food Classification Review":
            project = p
            logger.info(f"Found existing project: {project.title} (ID: {project.id})")
            break
            
    if project is None:
        logger.error("Food Classification Review project not found in Label Studio")
except Exception as e:
    logger.error(f"Error initializing Label Studio client: {e}")
    ls = None

def append_to_tracking_file(file_name, entry):
    """Add an entry to a tracking file"""
    file_path = f'{TRACKING_BUCKET}/{file_name}'
    
    try:
        # Read existing data
        if fs.exists(file_path):
            with fs.open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Append new entry
        data.append(entry)
        
        # Write back to file
        with fs.open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Added entry to {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error appending to tracking file {file_name}: {e}")
        return False

def get_all_images():
    """Get all images from the production-images bucket"""
    if not fs.exists(IMAGES_BUCKET):
        logger.error(f"Images bucket {IMAGES_BUCKET} does not exist")
        return []
    
    all_images = []
    try:
        # List all subdirectories (class_XX)
        class_dirs = [d for d in fs.ls(IMAGES_BUCKET) if fs.isdir(d)]
        
        # For each class directory, list all images
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            images = [f for f in fs.ls(class_dir) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
            
            for img_path in images:
                # Extract class number from directory name (class_XX)
                class_num = int(class_name.split('_')[1]) if class_name.startswith('class_') else -1
                # Get the filename (without the path)
                filename = os.path.basename(img_path)
                
                all_images.append({
                    'path': img_path,
                    'filename': filename,
                    'class_dir': class_dir,
                    'class_num': class_num
                })
        
        logger.info(f"Found {len(all_images)} total images across {len(class_dirs)} class directories")
        return all_images
    
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return []
    
def get_already_sampled_images():
    """Get list of all images that have already been sent to Label Studio for any reason"""
    # Check all tracking files to avoid re-sampling images already in Label Studio
    tracking_files = [
        'user_feedback_tasks.json',
        'low_confidence_tasks.json',
        'random_sampling_tasks.json'
    ]
    
    sampled_filenames = set()  # Use a set for more efficient lookups
    
    for file_name in tracking_files:
        file_path = f'{TRACKING_BUCKET}/{file_name}'
        
        if not fs.exists(file_path):
            continue
            
        try:
            with fs.open(file_path, 'r') as f:
                tasks = json.load(f)
            
            # Extract image filenames (not full paths) from tasks
            for task in tasks:
                # Extract the image path from the task
                image_path = task.get('image_path', '')
                if image_path:
                    # Get just the filename part
                    filename = os.path.basename(image_path)
                    sampled_filenames.add(filename)
                    
        except Exception as e:
            logger.error(f"Error reading tracking file {file_name}: {e}")
    
    logger.info(f"Found {len(sampled_filenames)} unique image filenames already sent to Label Studio")
    return sampled_filenames

def create_label_studio_task(image_path, class_num):
    """Create a task in Label Studio with the image"""
    if not ls or not project:
        logger.warning("Label Studio client or project not available. Skipping task creation.")
        return None
    
    # Generate the image URL that Label Studio can access
    image_url = urljoin(MINIO_PUBLIC_URL, image_path)
    
    # Get the class name from the class number
    classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", 
               "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", 
               "Vegetable/Fruit"]
    
    if 0 <= class_num < len(classes):
        predicted_class = classes[class_num]
    else:
        predicted_class = "Unknown"
    
    try:
        # Prepare task data according to Label Studio format
        task_data = {
            "image": image_url,
            "ml_prediction": predicted_class,
            "confidence": 1.0,  # Default high confidence since we're just doing random sampling
            "user_feedback": "random_sampling"  # Indicate this is from random sampling
        }
        
        # Create task in Label Studio
        task = ls.tasks.create(
            project=project.id, 
            data=task_data
        )
        
        logger.info(f"Task created successfully in Label Studio: {task.id}")
        
        # Create tracking entry
        tracking_entry = {
            "image_path": image_path,
            "original_prediction": predicted_class,
            "confidence": 1.0,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": "v1.0",
            "task_id": task.id,
            "status": "pending",
            "sampling_type": "random"
        }
        
        return tracking_entry
    
    except Exception as e:
        logger.error(f"Error creating Label Studio task: {e}")
        traceback.print_exc()
        return None

def sample_random_images(sample_count):
    """Sample random images from the production-images bucket"""
    # Get all images and already sampled images (filenames only)
    all_images = get_all_images()
    already_sampled_filenames = get_already_sampled_images()
    
    # Filter out already sampled images by filename
    available_images = [
        img for img in all_images 
        if img['filename'] not in already_sampled_filenames
    ]
    
    if not available_images:
        logger.warning("No images available for sampling")
        return []
    
    # Select random images
    sample_size = min(sample_count, len(available_images))
    sampled_images = random.sample(available_images, sample_size)
    logger.info(f"Randomly sampled {sample_size} images across all classes")
    
    # Create tasks for each sampled image
    new_tasks = []
    for img in sampled_images:
        tracking_entry = create_label_studio_task(img['path'], img['class_num'])
        if tracking_entry:
            new_tasks.append(tracking_entry)
    
    # Update tracking file with new tasks
    if new_tasks:
        for task in new_tasks:
            append_to_tracking_file('random_sampling_tasks.json', task)
    
    return new_tasks

def main():
    """Main function to run as a cron job"""
    try:
        logger.info(f"Starting random sampling process. Count: {SAMPLE_COUNT}")
        tasks = sample_random_images(SAMPLE_COUNT)
        logger.info(f"Completed sampling. Created {len(tasks)} new tasks.")
    except Exception as e:
        logger.error(f"Unexpected error in sampling process: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()