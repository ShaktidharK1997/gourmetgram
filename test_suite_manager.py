#!/usr/bin/env python3
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

class TestSuiteManager:
    """
    Creates test suites from different feedback sources by copying images
    to organized directories with timestamps.
    """
    
    TRACKING_FILES = {
        'user_feedback': 'user_feedback_tasks.json',
        'low_confidence': 'low_confidence_tasks.json',
        'random_sampling': 'random_sampling_tasks.json'
    }
    
    # List of supported food classes
    CLASSES = [
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"
    ]
    
    def __init__(self, storage_manager, logger=None):
        """
        Initialize with a storage manager instance
        """
        self.storage = storage_manager
        
        # Set up logger or use provided one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("test_suite_manager")
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
    
    def get_task_image_paths(self, task_type: str) -> List[str]:
        """
        Get paths to all images associated with a specific task type
        """
        if task_type not in self.TRACKING_FILES:
            self.logger.warning(f"Invalid task type: {task_type}")
            return []
            
        # Use the storage manager's method to get image paths
        return self.storage.get_task_image_paths(task_type)
    
    def create_test_suite(self, task_type: str) -> str:
        """
        Create a test suite for a specific task type with class subdirectories
        """
        if task_type not in self.TRACKING_FILES:
            self.logger.error(f"Invalid task type: {task_type}")
            return ""
        
        # Create timestamp for the test suite directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_suite_dir = f"{self.storage.TEST_SUITES_BUCKET}/{task_type}/{timestamp}"
        
        # Create the test suite directory
        if not self.storage.fs.exists(test_suite_dir):
            self.storage.fs.mkdir(test_suite_dir)
            self.logger.info(f"Created test suite directory: {test_suite_dir}")
        
        # Get image paths for the task type
        image_paths = self.get_task_image_paths(task_type)
        if not image_paths:
            self.logger.warning(f"No images found for task type: {task_type}")
            return test_suite_dir
        
        # Create class subdirectories
        class_dirs = {}
        for i in range(len(self.CLASSES)):
            class_dir_path = f"{test_suite_dir}/class_{i:02d}"
            if not self.storage.fs.exists(class_dir_path):
                self.storage.fs.mkdir(class_dir_path)
            class_dirs[i] = class_dir_path
        
        # Copy images to the appropriate class subdirectory
        copied_count = 0
        image_stats = {}  # Track images per class
        
        for source_path in image_paths:
            try:
                # Extract the class from the source path
                # Source path format is usually: 'production-images/class_XX/filename.jpg'
                path_parts = source_path.split('/')
                
                # Find the part that contains class_XX
                class_part = None
                for part in path_parts:
                    if part.startswith('class_'):
                        class_part = part
                        break
                
                if class_part:
                    # Extract class index from the directory name
                    try:
                        class_idx = int(class_part.split('_')[1])
                        class_dir = class_dirs[class_idx]
                    except (ValueError, IndexError, KeyError):
                        # If parsing fails, use a default "unknown" directory
                        unknown_dir = f"{test_suite_dir}/unknown"
                        if not self.storage.fs.exists(unknown_dir):
                            self.storage.fs.mkdir(unknown_dir)
                        class_dir = unknown_dir
                else:
                    # If no class directory found, use "unknown"
                    unknown_dir = f"{test_suite_dir}/unknown"
                    if not self.storage.fs.exists(unknown_dir):
                        self.storage.fs.mkdir(unknown_dir)
                    class_dir = unknown_dir
                
                # Create target path with original filename in the appropriate class directory
                filename = Path(source_path).name
                target_path = f"{class_dir}/{filename}"
                
                # Copy the file
                self.storage.fs.copy(source_path, target_path)
                copied_count += 1
                
                # Track stats
                class_name = class_part if class_part else "unknown"
                if class_name not in image_stats:
                    image_stats[class_name] = 0
                image_stats[class_name] += 1
                
            except Exception as e:
                self.logger.error(f"Error copying {source_path}: {e}")
        
        self.logger.info(f"Copied {copied_count} of {len(image_paths)} images to {test_suite_dir}")
        for class_name, count in image_stats.items():
            self.logger.info(f"  - {class_name}: {count} images")
        
        # Create a metadata file with information about the test suite
        metadata = {
            "task_type": task_type,
            "created_at": datetime.datetime.now().isoformat(),
            "image_count": copied_count,
            "source_tracking_file": self.TRACKING_FILES[task_type],
            "class_distribution": image_stats
        }
        
        try:
            metadata_path = f"{test_suite_dir}/metadata.json"
            self.storage.write_json_file(metadata_path, metadata)
            self.logger.info(f"Created metadata file at {metadata_path}")
        except Exception as e:
            self.logger.error(f"Error creating metadata file: {e}")
        
        return test_suite_dir
    
    def create_all_test_suites(self) -> Dict[str, str]:
        """
        Create test suites for all task types
        
        Returns:
            Dictionary mapping task types to their test suite directories
        """
        results = {}
        for task_type in self.TRACKING_FILES.keys():
            self.logger.info(f"Creating test suite for {task_type}")
            test_suite_dir = self.create_test_suite(task_type)
            results[task_type] = test_suite_dir
        return results