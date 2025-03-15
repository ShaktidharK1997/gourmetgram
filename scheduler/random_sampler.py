#!/usr/bin/env python3
import os
import json
import random
import s3fs
import logging
import traceback
import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from urllib.parse import urljoin
from label_studio_sdk.client import LabelStudio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RandomSampler:
    """Class to handle random sampling of images for Label Studio"""
    
    CLASSES = [
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"
    ]
    
    IMAGES_BUCKET = 'production-images'
    TRACKING_BUCKET = 'tracking'
    
    # Tracking files
    TRACKING_FILES = [
        'user_feedback_tasks.json',
        'low_confidence_tasks.json',
        'random_sampling_tasks.json'
    ]
    
    def __init__(
        self, 
        minio_endpoint: str = 'http://minio:9000',
        minio_public_url: str = 'http://localhost:9000',
        minio_key: str = 'minioadmin',
        minio_secret: str = 'minioadmin',
        label_studio_url: str = 'http://label-studio:8080',
        label_studio_token: Optional[str] = None,
        sample_count: int = 3,
        log_file: str = '/var/log/random_sampler.log'
    ):
        """Initialize the sampler with connection details"""
        # Set up logging
        self._setup_logging(log_file)
        
        # Store configuration
        self.minio_endpoint = minio_endpoint
        self.minio_public_url = minio_public_url
        self.sample_count = sample_count
        
        # Initialize S3 filesystem
        self._init_s3_filesystem(minio_key, minio_secret, minio_endpoint)
        
        # Initialize Label Studio client
        self._init_label_studio(label_studio_url, label_studio_token)
    
    def _setup_logging(self, log_file: str) -> None:
        """Set up logging configuration"""
        self.logger = logging.getLogger("random_sampler")
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _init_s3_filesystem(self, key: str, secret: str, endpoint: str) -> None:
        """Initialize the S3 filesystem connection"""
        try:
            self.fs = s3fs.S3FileSystem(
                key=key,
                secret=secret,
                client_kwargs={'endpoint_url': endpoint}
            )
            self.logger.info("S3 filesystem initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 filesystem: {e}")
            raise
    
    def _init_label_studio(self, url: str, token: Optional[str]) -> None:
        """Initialize the Label Studio client and find the project"""
        self.ls = None
        self.project = None
        
        if not token:
            self.logger.warning("No Label Studio token provided")
            return
            
        try:
            self.ls = LabelStudio(
                base_url=url,
                api_key=token
            )
            self.logger.info("Label Studio client initialized successfully")
            
            projects = self.ls.projects.list()
            for p in projects:
                if p.title == "Food Classification Review":
                    self.project = p
                    self.logger.info(f"Found existing project: {self.project.title} (ID: {self.project.id})")
                    break
                    
            if self.project is None:
                self.logger.error("Food Classification Review project not found in Label Studio")
        except Exception as e:
            self.logger.error(f"Error initializing Label Studio client: {e}")
            self.ls = None
    
    def _read_json_file(self, file_path: str) -> List[Dict]:
        """Read a JSON file from S3 with error handling"""
        try:
            with self.fs.open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in file: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def _write_json_file(self, file_path: str, data: Any) -> bool:
        """Write a JSON file to S3 with error handling"""
        try:
            with self.fs.open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error writing to {file_path}: {e}")
            return False
    
    def append_to_tracking_file(self, file_name: str, entry: Dict) -> bool:
        """Add an entry to a tracking file"""
        file_path = f'{self.TRACKING_BUCKET}/{file_name}'
        
        try:
            if self.fs.exists(file_path):
                data = self._read_json_file(file_path)
            else:
                data = []
            
            data.append(entry)
            
            success = self._write_json_file(file_path, data)
            if success:
                self.logger.info(f"Added entry to {file_name}")
            return success
        except Exception as e:
            self.logger.error(f"Error appending to tracking file {file_name}: {e}")
            return False
    
    def get_all_images(self) -> List[Dict]:
        """Get all images from the production-images bucket"""
        if not self.fs.exists(self.IMAGES_BUCKET):
            self.logger.error(f"Images bucket {self.IMAGES_BUCKET} does not exist")
            return []
        
        all_images = []
        try:
            class_dirs = [d for d in self.fs.ls(self.IMAGES_BUCKET) if self.fs.isdir(d)]
            
            for class_dir in class_dirs:
                class_name = Path(class_dir).name
                images = [
                    f for f in self.fs.ls(class_dir) 
                    if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png']
                ]
                
                for img_path in images:
                    # Extract class number from directory name (class_XX)
                    class_num = int(class_name.split('_')[1]) if class_name.startswith('class_') else -1
                    # Get the filename (without the path)
                    filename = Path(img_path).name
                    
                    all_images.append({
                        'path': img_path,
                        'filename': filename,
                        'class_dir': class_dir,
                        'class_num': class_num
                    })
            
            self.logger.info(f"Found {len(all_images)} total images across {len(class_dirs)} class directories")
            return all_images
        
        except Exception as e:
            self.logger.error(f"Error listing images: {e}")
            return []
    
    def get_already_sampled_images(self) -> Set[str]:
        """Get list of all images that have already been sent to Label Studio for any reason"""
        sampled_filenames = set()  
        
        for file_name in self.TRACKING_FILES:
            file_path = f'{self.TRACKING_BUCKET}/{file_name}'
            
            if not self.fs.exists(file_path):
                continue
                
            try:
                tasks = self._read_json_file(file_path)
                
                # Extract image filenames (not full paths) from tasks
                for task in tasks:
                    # Extract the image path from the task
                    image_path = task.get('image_path', '')
                    if image_path:
                        # Get just the filename part
                        filename = Path(image_path).name
                        sampled_filenames.add(filename)
                        
            except Exception as e:
                self.logger.error(f"Error reading tracking file {file_name}: {e}")
        
        self.logger.info(f"Found {len(sampled_filenames)} unique image filenames already sent to Label Studio")
        return sampled_filenames
    
    def create_label_studio_task(self, image_path: str, class_num: int) -> Optional[Dict]:
        """Create a task in Label Studio with the image"""
        if not self.ls or not self.project:
            self.logger.warning("Label Studio client or project not available. Skipping task creation.")
            return None
        
        # Generate the image URL that Label Studio can access
        image_url = urljoin(self.minio_public_url, image_path)
        
        # Get the class name from the class number
        if 0 <= class_num < len(self.CLASSES):
            predicted_class = self.CLASSES[class_num]
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
            task = self.ls.tasks.create(
                project=self.project.id, 
                data=task_data
            )
            
            self.logger.info(f"Task created successfully in Label Studio: {task.id}")
            
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
            self.logger.error(f"Error creating Label Studio task: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def sample_random_images(self) -> List[Dict]:
        """Sample random images from the production-images bucket"""
        # Get all images and already sampled images (filenames only)
        all_images = self.get_all_images()
        already_sampled_filenames = self.get_already_sampled_images()
        
        # Filter out already sampled images by filename
        available_images = [
            img for img in all_images 
            if img['filename'] not in already_sampled_filenames
        ]
        
        if not available_images:
            self.logger.warning("No images available for sampling")
            return []
        
        # Select random images
        sample_size = min(self.sample_count, len(available_images))
        sampled_images = random.sample(available_images, sample_size)
        self.logger.info(f"Randomly sampled {sample_size} images across all classes")
        
        # Create tasks for each sampled image
        new_tasks = []
        for img in sampled_images:
            tracking_entry = self.create_label_studio_task(img['path'], img['class_num'])
            if tracking_entry:
                new_tasks.append(tracking_entry)
        
        # Update tracking file with new tasks
        if new_tasks:
            for task in new_tasks:
                self.append_to_tracking_file('random_sampling_tasks.json', task)
        
        return new_tasks
    
    def run(self) -> int:
        """Main method to execute the sampling process"""
        try:
            self.logger.info(f"Starting random sampling process. Count: {self.sample_count}")
            tasks = self.sample_random_images()
            self.logger.info(f"Completed sampling. Created {len(tasks)} new tasks.")
            return len(tasks)
        except Exception as e:
            self.logger.error(f"Unexpected error in sampling process: {e}")
            self.logger.error(traceback.format_exc())
            return 0


def main():
    """Main function to run as a cron job"""

    minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'http://minio:9000')
    minio_public_url = os.environ.get('MINIO_PUBLIC_URL', 'http://localhost:9000')
    minio_key = os.environ.get('MINIO_ROOT_USER', 'minioadmin')
    minio_secret = os.environ.get('MINIO_ROOT_PASSWORD', 'minioadmin')
    label_studio_url = os.environ.get('LABEL_STUDIO_URL', 'http://label-studio:8080')
    label_studio_token = os.environ.get('LABEL_STUDIO_USER_TOKEN', "ab9927067c51ff279d340d7321e4890dc2841c4a")
    sample_count = int(os.environ.get('SAMPLE_COUNT', '3'))
    log_file = '/var/log/random_sampler.log'
    
    # Create and run the sampler
    sampler = RandomSampler(
        minio_endpoint=minio_endpoint,
        minio_public_url=minio_public_url,
        minio_key=minio_key,
        minio_secret=minio_secret,
        label_studio_url=label_studio_url,
        label_studio_token=label_studio_token,
        sample_count=sample_count,
        log_file=log_file
    )
    
    sampler.run()


if __name__ == "__main__":
    main()
