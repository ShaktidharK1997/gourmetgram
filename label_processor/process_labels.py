import os
import json
import s3fs
import logging
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(
    key='minioadmin', 
    secret='minioadmin',
    client_kwargs={
        'endpoint_url': 'http://minio:9000'
    }
)

# Constants
TARGET_BUCKET = 'target-bucket'
TRACKING_BUCKET = 'tracking'
IMAGES_BUCKET = 'production-images'
PROCESSED_FILES_TRACKER = f'{TRACKING_BUCKET}/processed_label_files.json'
TRACKING_FILES = [
    'user_feedback_tasks.json',
    'low_confidence_tasks.json',
    'random_sampling_tasks.json'
]

# Define the classes array for mapping
CLASSES = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"])

# Ensure processed files tracker exists
if not fs.exists(PROCESSED_FILES_TRACKER):
    with fs.open(PROCESSED_FILES_TRACKER, 'w') as f:
        json.dump([], f)

def get_class_directory_from_label(label):
    """Convert a class label to the corresponding directory name"""
    try:
        class_index = np.where(CLASSES == label)[0][0]
        return f"class_{class_index:02d}"
    except (IndexError, ValueError):
        logger.error(f"Could not find index for label: {label}")
        return None

def get_processed_files():
    """Get list of already processed files"""
    try:
        with fs.open(PROCESSED_FILES_TRACKER, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading processed files tracker: {e}")
        return []

def add_processed_file(file_path):
    """Add a file to the processed files tracker"""
    try:
        processed_files = get_processed_files()
        processed_files.append(file_path)
        with fs.open(PROCESSED_FILES_TRACKER, 'w') as f:
            json.dump(processed_files, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error updating processed files tracker: {e}")
        return False

def update_tracking_status(task_id, new_label):
    """Update the status of a task in the tracking files"""
    for file_name in TRACKING_FILES:
        file_path = f'{TRACKING_BUCKET}/{file_name}'
        
        if not fs.exists(file_path):
            continue
            
        try:
            # Read file
            with fs.open(file_path, 'r') as f:
                data = json.load(f)
            
            # Update entry if found
            updated = False
            for entry in data:
                if entry.get('task_id') == task_id:
                    entry['status'] = 'labeled'
                    entry['final_label'] = new_label
                    entry['label_timestamp'] = datetime.now().isoformat()
                    updated = True
                    break
            
            # Write back if updated
            if updated:
                with fs.open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Updated task {task_id} status in {file_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating tracking file {file_name}: {e}")
    
    return False

def process_label_studio_results():
    """Process the labeled tasks in the target bucket"""
    try:
        # Get already processed files
        processed_files = get_processed_files()
        
        # List all JSON files in the target bucket
        json_files = [f for f in fs.ls(TARGET_BUCKET) if f not in processed_files]
        logger.info(f"Found {len(json_files)} new JSON files to process")
        
        for json_path in json_files:
            try:
                # Read the JSON file
                with fs.open(json_path, 'r') as f:
                    task_data = json.load(f)
                
                # Extract relevant information
                task_id = task_data.get('id')
                label_result = task_data.get('result', [])
                
                if not label_result:
                    logger.warning(f"No result found in {json_path}")
                    continue
                
                # Get the new label from human annotator
                new_label = label_result[0].get('value', {}).get('choices', ['Unknown'])[0]
                new_class_dir = get_class_directory_from_label(new_label)
                
                if not new_class_dir:
                    logger.warning(f"Could not determine class directory for label: {new_label}")
                    continue
                
                # Get the image path from task data
                image_url = task_data.get('task', {}).get('data', {}).get('image', '')
                if not image_url:
                    logger.warning(f"No image URL found in {json_path}")
                    continue
                
                # Extract full path from URL - handle the class subdirectory structure
                image_path = image_url.replace('http://localhost:9000/', '')
                
                # Move to the new class directory
                if fs.exists(image_path):
                    filename = image_path.split('/')[-1]
                    target_path = f"{IMAGES_BUCKET}/{new_class_dir}/{filename}"
                    
                    # Only move if source and target are different
                    if image_path != target_path:
                        fs.copy(image_path, target_path)
                        logger.info(f"Moved {image_path} to {target_path}")
                        
                        # Optionally delete the original if needed
                        fs.rm(image_path)
                    else:
                        logger.info(f"Image already in correct directory: {image_path}")
                    
                    # Update tracking status
                    update_tracking_status(task_id, new_label)
                else:
                    logger.warning(f"Source image not found: {image_path}")
                
                # Mark file as processed
                add_processed_file(json_path)
                logger.info(f"Processed {json_path}")
                
                # Delete the processed JSON file after tracking it
                fs.rm(json_path)
                logger.info(f"Deleted processed file: {json_path}")
                
            except Exception as e:
                logger.error(f"Error processing {json_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error listing JSON files: {e}")

if __name__ == "__main__":
    process_label_studio_results()