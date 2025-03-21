import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path

class LabelProcessor:
    """
    Processes Label Studio annotation results and organizes images based on labels.
    
    This class handles:
    - Reading and processing Label Studio annotation results
    - Moving images to the appropriate class folders
    - Tracking processed files
    - Updating task statuses in tracking files
    """
    
    # Classes mapping
    CLASSES = [
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"
    ]
    
    def __init__(self, storage_manager, logger=None):
        """
        Initialize the processor with a storage manager instance
        
        Args:
            storage_manager: Instance that provides S3/MinIO access
            logger: Logger instance (optional)
        """
        self.storage = storage_manager
        self.fs = storage_manager.fs
        
        # Set up logger or use provided one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("label_processor")
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                
        # Bucket paths
        self.TARGET_BUCKET = 'target-bucket'
        self.TRACKING_BUCKET = self.storage.TRACKING_BUCKET
        self.IMAGES_BUCKET = self.storage.BUCKET_NAME
        
        # Processed files tracker
        self.PROCESSED_FILES_TRACKER = f'{self.TRACKING_BUCKET}/processed_label_files.json'
        
        # Tracking files
        self.TRACKING_FILES = [
            'user_feedback_tasks.json',
            'low_confidence_tasks.json',
            'random_sampling_tasks.json'
        ]
        
        # Ensure processed files tracker exists
        self._initialize_tracker()
    
    def _initialize_tracker(self) -> None:
        """Ensure the processed files tracker exists"""
        if not self.fs.exists(self.PROCESSED_FILES_TRACKER):
            self.storage.write_json_file(self.PROCESSED_FILES_TRACKER, [])
            self.logger.info(f"Created new processed files tracker at {self.PROCESSED_FILES_TRACKER}")
    
    def get_class_directory(self, label: str) -> Optional[str]:
        """Get the correct class directory for a label"""
        try:
            class_index = self.CLASSES.index(label)
            return f"class_{class_index:02d}"
        except ValueError:
            self.logger.error(f"Invalid label: {label}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting class directory for {label}: {e}")
            return None

    def get_processed_files(self) -> List[str]:
        """Get list of already processed files"""
        data = self.storage.read_json_file(self.PROCESSED_FILES_TRACKER)
        return data if data is not None else []
    
    def add_processed_file(self, file_path: str) -> bool:
        """Add a file to the processed files tracker"""
        processed_files = self.get_processed_files()
        if file_path not in processed_files:
            processed_files.append(file_path)
            return self.storage.write_json_file(self.PROCESSED_FILES_TRACKER, processed_files)
        return True
    
    def update_tracking_status(self, task_id: str, new_label: str) -> bool:
        """
        Update the status of a task in the tracking files
        """
        for file_name in self.TRACKING_FILES:
            file_path = f'{self.TRACKING_BUCKET}/{file_name}'
            
            if not self.fs.exists(file_path):
                continue
                
            data = self.storage.read_json_file(file_path)
            if data is None:
                continue
            
            # Update entry if found
            for entry in data:
                if entry.get('task_id') == task_id:
                    entry.update({
                        'status': 'labeled',
                        'final_label': new_label,
                        'label_timestamp': datetime.now().isoformat()
                    })
                    
                    # Write back the updated data
                    if self.storage.write_json_file(file_path, data):
                        self.logger.info(f"Updated task {task_id} status in {file_name}")
                        return True
        
        self.logger.warning(f"Could not find task {task_id} in any tracking file")
        return False
    
    def extract_task_info(self, task_data: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract key information from a task data object
        """
        # Extract task ID
        task_id = task_data.get('id')
        if not task_id:
            self.logger.warning("No task ID found in task data")
            return None, None, None
        
        # Extract label from result
        label_result = task_data.get('result', [])
        if not label_result:
            self.logger.warning(f"No result found for task {task_id}")
            return task_id, None, None
        
        # Get the new label from human annotator
        new_label = None
        try:
            new_label = label_result[0].get('value', {}).get('choices', ['Unknown'])[0]
            if new_label == 'Unknown':
                self.logger.warning(f"Could not extract valid label for task {task_id}")
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error extracting label for task {task_id}: {e}")
            return task_id, None, None
        
        # Get image path
        image_url = task_data.get('task', {}).get('data', {}).get('image', '')
        if not image_url:
            self.logger.warning(f"No image URL found for task {task_id}")
            return task_id, new_label, None
        
        # Extract full path from URL
        image_path = image_url.replace('http://localhost:9000/', '')
        
        return task_id, new_label, image_path
    
    def move_image_to_class_directory(self, image_path: str, new_label: str) -> bool:
        """
        Move an image to the appropriate class directory
        """
        # Get the class directory
        new_class_dir = self.get_class_directory(new_label)
        if not new_class_dir:
            return False
        
        # Check if the source image exists
        if not self.fs.exists(image_path):
            self.logger.warning(f"Source image not found: {image_path}")
            return False
        
        # Move to the new class directory
        filename = Path(image_path).name
        target_path = f"{self.IMAGES_BUCKET}/{new_class_dir}/{filename}"
        
        # Only move if source and target are different
        if image_path != target_path:
            try:
                self.fs.copy(image_path, target_path)
                self.logger.info(f"Moved {image_path} to {target_path}")
                
                # Delete the original
                self.fs.rm(image_path)
                self.logger.info(f"Deleted original image: {image_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error moving image {image_path}: {e}")
                return False
        else:
            self.logger.info(f"Image already in correct directory: {image_path}")
            return True
    
    def process_single_file(self, json_path: str) -> bool:
        """
        Process a single Label Studio result file
        """
        # Read the task data
        task_data = self.storage.read_json_file(json_path)
        if task_data is None:
            return False
        
        # Extract task information
        task_id, new_label, image_path = self.extract_task_info(task_data)
        if not all([task_id, new_label, image_path]):
            self.logger.warning(f"Missing required information to process {json_path}")
            return False
        
        # Move the image to the correct directory
        if self.move_image_to_class_directory(image_path, new_label):
            # Update tracking status
            self.update_tracking_status(task_id, new_label)
            
            # Mark file as processed
            if self.add_processed_file(json_path):
                self.logger.info(f"Marked {json_path} as processed")
                return True
        
        return False
        
    def process_label_studio_results(self) -> Dict[str, Any]:
        """
        Process all labeled tasks in the target bucket
        """
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        processed_files = []
        failed_files = []
        
        # Get already processed files
        already_processed = self.get_processed_files()

        # Get all JSON files in the target bucket
        try:
            all_files = self.fs.ls(self.TARGET_BUCKET)
            json_files = [f for f in all_files if f not in already_processed]
        except Exception as e:
            self.logger.error(f"Error listing files in {self.TARGET_BUCKET}: {e}")
            return {
                "status": "error",
                "message": f"Failed to list files: {str(e)}",
                "processed_count": 0,
                "failed_count": 0,
                "skipped_count": len(already_processed)
            }
        
        # Process each file
        for json_path in json_files:
            if json_path in already_processed:
                skipped_count += 1
                continue
                
            if self.process_single_file(json_path):
                processed_count += 1
                processed_files.append(json_path)
            else:
                failed_count += 1
                failed_files.append(json_path)
        
        return {
            "status": "success",
            "message": f"Processed {processed_count} files, failed {failed_count}, skipped {skipped_count}",
            "processed_count": processed_count,
            "failed_count": failed_count, 
            "skipped_count": skipped_count,
            "processed_files": processed_files,
            "failed_files": failed_files
        }