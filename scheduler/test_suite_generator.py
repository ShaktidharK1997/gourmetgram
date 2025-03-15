#!/usr/bin/env python3
import os
import argparse
import json
import s3fs
import logging
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Set, Optional

class TestSuiteGenerator:
    """
    Creates test suites from different feedback sources by copying images
    to organized directories with timestamps.
    """
    
    # Bucket paths
    IMAGES_BUCKET = 'production-images'
    TRACKING_BUCKET = 'tracking'
    TEST_SUITES_BUCKET = 'test-suites'
    
    TEST_SUITE_DIRS = ['user_feedback', 'low_confidence', 'random_sampling']
    
    # Tracking files that correspond to test suite types
    TRACKING_FILES = {
        'user_feedback': 'user_feedback_tasks.json',
        'low_confidence': 'low_confidence_tasks.json',
        'random_sampling': 'random_sampling_tasks.json'
    }
    
    def __init__(
        self, 
        minio_endpoint: str = os.environ.get('MINIO_ENDPOINT'),
        minio_key: str = os.environ.get('MINIO_ROOT_USER'),
        minio_secret: str = os.environ.get('MINIO_ROOT_PASSWORD'),
        log_file: str = '/var/log/test_suite_generator.log'
    ):
        """Initialize the test suite generator with storage connection details"""
        # Set up logging
        self._setup_logging(log_file)
        
        # Initialize S3 filesystem
        self._init_s3_filesystem(minio_key, minio_secret, minio_endpoint)
        
        # Set up test suites bucket and structure
        self._setup_test_suites_structure()
    
    def _setup_logging(self, log_file: str) -> None:
        """Set up logging configuration"""
        self.logger = logging.getLogger("test_suite_generator")
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # File handler
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
    
    def _setup_test_suites_structure(self) -> None:
        """Create test-suites bucket and subdirectories if they don't exist"""
        try:
            # Create test-suites bucket if it doesn't exist
            if not self.fs.exists(self.TEST_SUITES_BUCKET):
                self.fs.mkdir(self.TEST_SUITES_BUCKET)
                self.logger.info(f"Created bucket: {self.TEST_SUITES_BUCKET}")
            
            # Create subdirectories for each test suite type
            for dir_name in self.TEST_SUITE_DIRS:
                test_suite_dir = f"{self.TEST_SUITES_BUCKET}/{dir_name}"
                if not self.fs.exists(test_suite_dir):
                    self.fs.mkdir(test_suite_dir)
                    self.logger.info(f"Created test suite directory: {test_suite_dir}")
                    
        except Exception as e:
            self.logger.error(f"Error setting up test suites structure: {e}")
            raise
    
    def _read_json_file(self, file_path: str) -> List[Dict]:
        """Read a JSON file from S3 with error handling"""
        try:
            with self.fs.open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in file: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def _get_task_image_paths(self, task_type: str) -> List[str]:
        """Get paths to all images associated with a specific task type"""
        
        tracking_file = f"{self.TRACKING_BUCKET}/{self.TRACKING_FILES[task_type]}"
        if not self.fs.exists(tracking_file):
            self.logger.warning(f"Tracking file not found: {tracking_file}")
            return []
        
        image_paths = []
        tasks = self._read_json_file(tracking_file)
        
        # CLASSES mapping to use for finding the correct directory
        CLASSES = [
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ]
        
        for task in tasks:
            # Get image path from task data
            image_path = task.get('image_path', '')
            if image_path and self.fs.exists(image_path):
                image_paths.append(image_path)
            else:
                # If original path not found, check if it's a labeled task with a final label
                final_label = task.get('final_label')
                status = task.get('status')
                
                if status == 'labeled' and final_label and final_label in CLASSES:
                    # Get the class index based on the final label
                    class_idx = CLASSES.index(final_label)
                    
                    # Extract filename from the original path
                    filename = Path(image_path).name
                    
                    # Go directly to the correct class directory
                    potential_path = f"{self.IMAGES_BUCKET}/class_{class_idx:02d}/{filename}"
                    if self.fs.exists(potential_path):
                        image_paths.append(potential_path)
                        self.logger.info(f"Found reclassified image at {potential_path}")
        
        self.logger.info(f"Found {len(image_paths)} images for task type: {task_type}")
        return image_paths
    
    def create_test_suite(self, task_type: str) -> str:
        """Create a test suite for a specific task type"""
        if task_type not in self.TRACKING_FILES:
            self.logger.error(f"Invalid task type: {task_type}")
            return ""
        
        # Create timestamp for the test suite directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_suite_dir = f"{self.TEST_SUITES_BUCKET}/{task_type}/{timestamp}"
        
        # Create the directory
        if not self.fs.exists(test_suite_dir):
            self.fs.mkdir(test_suite_dir)
            self.logger.info(f"Created test suite directory: {test_suite_dir}")
        
        # Get image paths for the task type
        image_paths = self._get_task_image_paths(task_type)
        if not image_paths:
            self.logger.warning(f"No images found for task type: {task_type}")
            return test_suite_dir
        
        # Copy images to the test suite directory
        copied_count = 0
        for source_path in image_paths:
            try:
                # Create target path with original filename
                filename = Path(source_path).name
                target_path = f"{test_suite_dir}/{filename}"
                
                # Copy the file
                self.fs.copy(source_path, target_path)
                copied_count += 1
                
            except Exception as e:
                self.logger.error(f"Error copying {source_path}: {e}")
        
        self.logger.info(f"Copied {copied_count} of {len(image_paths)} images to {test_suite_dir}")
        
        # Create a metadata file with information about the test suite
        metadata = {
            "task_type": task_type,
            "created_at": datetime.datetime.now().isoformat(),
            "image_count": copied_count,
            "source_tracking_file": self.TRACKING_FILES[task_type]
        }
        
        try:
            metadata_path = f"{test_suite_dir}/metadata.json"
            with self.fs.open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Created metadata file at {metadata_path}")
        except Exception as e:
            self.logger.error(f"Error creating metadata file: {e}")
        
        return test_suite_dir
    
    def create_all_test_suites(self) -> Dict[str, str]:
        """Create test suites for all task types"""
        results = {}
        for task_type in self.TRACKING_FILES.keys():
            self.logger.info(f"Creating test suite for {task_type}")
            test_suite_dir = self.create_test_suite(task_type)
            results[task_type] = test_suite_dir
        return results


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate test suites from feedback data')
    parser.add_argument(
        'task_type', 
        choices=['user_feedback', 'low_confidence', 'random_sampling', 'all'],
        help='Type of task to create test suite for'
    )
    
    # Optional log file parameter
    parser.add_argument('--log-file', 
                       default='/var/log/test_suite_generator.log',
                       help='Path to log file')
    
    args = parser.parse_args()
    
    try:
        # Initialize the generator with environment variables
        generator = TestSuiteGenerator(
            log_file=args.log_file
        )
        
        # Generate test suites based on the task type
        if args.task_type == 'all':
            results = generator.create_all_test_suites()
            for task_type, test_suite_dir in results.items():
                print(f"Created test suite for {task_type} at {test_suite_dir}")
        else:
            test_suite_dir = generator.create_test_suite(args.task_type)
            print(f"Created test suite for {args.task_type} at {test_suite_dir}")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())