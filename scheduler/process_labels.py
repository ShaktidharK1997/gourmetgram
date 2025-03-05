import os
import json
import s3fs
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelProcessor:
    """Class to handle Label Studio result processing"""
    
    # Classes mapping
    CLASSES = np.array([
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"
    ])
    
    # Bucket paths
    TARGET_BUCKET = 'target-bucket'
    TRACKING_BUCKET = 'tracking'
    IMAGES_BUCKET = 'production-images'
    
    # Tracking files
    PROCESSED_FILES_TRACKER = f'{TRACKING_BUCKET}/processed_label_files.json'
    TRACKING_FILES = [
        'user_feedback_tasks.json',
        'low_confidence_tasks.json',
        'random_sampling_tasks.json'
    ]
    
    def __init__(self, endpoint_url: str = "http://minio:9000", 
                 access_key: str = "minioadmin", 
                 secret_key: str = "minioadmin"):
        """Initialize the processor with S3 connection details"""
        self.fs = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            client_kwargs={'endpoint_url': endpoint_url}
        )
        
        # Ensure processed files tracker exists
        if not self.fs.exists(self.PROCESSED_FILES_TRACKER):
            self._write_json_file(self.PROCESSED_FILES_TRACKER, [])
            logger.info(f"Created new processed files tracker at {self.PROCESSED_FILES_TRACKER}")
    
    def _read_json_file(self, file_path: str) -> Optional[Any]:
        """Read a JSON file from S3"""
        try:
            with self.fs.open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _write_json_file(self, file_path: str, data: Any) -> bool:
        """Write a JSON file to S3"""
        try:
            with self.fs.open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return False
    
    def get_class_directory(self, label: str) -> Optional[str]:
        """Convert a class label to the corresponding directory name"""
        try:
            class_index = np.where(self.CLASSES == label)[0][0]
            return f"class_{class_index:02d}"
        except (IndexError, ValueError):
            logger.error(f"Could not find index for label: {label}")
            return None
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed files"""
        data = self._read_json_file(self.PROCESSED_FILES_TRACKER)
        return data if data is not None else []
    
    def add_processed_file(self, file_path: str) -> bool:
        """Add a file to the processed files tracker"""
        processed_files = self.get_processed_files()
        processed_files.append(file_path)
        return self._write_json_file(self.PROCESSED_FILES_TRACKER, processed_files)
    
    def update_tracking_status(self, task_id: str, new_label: str) -> bool:
        """Update the status of a task in the tracking files"""
        for file_name in self.TRACKING_FILES:
            file_path = f'{self.TRACKING_BUCKET}/{file_name}'
            
            if not self.fs.exists(file_path):
                continue
                
            data = self._read_json_file(file_path)
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
                    if self._write_json_file(file_path, data):
                        logger.info(f"Updated task {task_id} status in {file_name}")
                        return True
        
        logger.warning(f"Could not find task {task_id} in any tracking file")
        return False
    
    def extract_task_info(self, task_data: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract key information from a task data object"""
        # Extract task ID
        task_id = task_data.get('id')
        if not task_id:
            logger.warning("No task ID found in task data")
            return None, None, None
        
        # Extract label from result
        label_result = task_data.get('result', [])
        if not label_result:
            logger.warning(f"No result found for task {task_id}")
            return task_id, None, None
        
        # Get the new label from human annotator
        new_label = label_result[0].get('value', {}).get('choices', ['Unknown'])[0]
        if new_label == 'Unknown':
            logger.warning(f"Could not extract valid label for task {task_id}")
        
        # Get image path
        image_url = task_data.get('task', {}).get('data', {}).get('image', '')
        if not image_url:
            logger.warning(f"No image URL found for task {task_id}")
            return task_id, new_label, None
        
        # Extract full path from URL
        image_path = image_url.replace('http://localhost:9000/', '')
        
        return task_id, new_label, image_path
    
    def move_image_to_class_directory(self, image_path: str, new_label: str) -> bool:
        """Move an image to the appropriate class directory"""
        # Get the class directory
        new_class_dir = self.get_class_directory(new_label)
        
        # Check if the source image exists
        if not self.fs.exists(image_path):
            logger.warning(f"Source image not found: {image_path}")
            return False
        
        # Move to the new class directory
        filename = Path(image_path).name
        target_path = f"{self.IMAGES_BUCKET}/{new_class_dir}/{filename}"
        
        # Only move if source and target are different
        if image_path != target_path:
            try:
                self.fs.copy(image_path, target_path)
                logger.info(f"Moved {image_path} to {target_path}")
                
                # Delete the original
                self.fs.rm(image_path)
                logger.info(f"Deleted original image: {image_path}")
                return True
            except Exception as e:
                logger.error(f"Error moving image {image_path}: {e}")
                return False
        else:
            logger.info(f"Image already in correct directory: {image_path}")
            return True
    
    def process_single_file(self, json_path: str) -> bool:
        """Process a single Label Studio result file"""
        # Read the task data
        task_data = self._read_json_file(json_path)
        if task_data is None:
            return False
        
        # Extract task information
        task_id, new_label, image_path = self.extract_task_info(task_data)
        if not all([task_id, new_label, image_path]):
            logger.warning(f"Missing required information to process {json_path}")
            return False
        
        # Move the image to the correct directory
        if self.move_image_to_class_directory(image_path, new_label):
            # Update tracking status
            self.update_tracking_status(task_id, new_label)
            
            # Mark file as processed
            if self.add_processed_file(json_path):
                logger.info(f"Marked {json_path} as processed")
                return True
        
        return False
        
    def process_label_studio_results(self) -> int:
        """Process all labeled tasks in the target bucket"""
        processed_count = 0
        
        try:
            # Get already processed files
            processed_files = self.get_processed_files()
            
            # List all files in the target bucket
            try:
                all_files = self.fs.ls(self.TARGET_BUCKET)
                json_files = [f for f in all_files if f not in processed_files]
                logger.info(f"Found {len(json_files)} new JSON files to process")
            except Exception as e:
                logger.error(f"Error listing files in {self.TARGET_BUCKET}: {e}")
                return processed_count
            
            # Process each file
            for json_path in json_files:
                if self.process_single_file(json_path):
                    processed_count += 1
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Unexpected error in processing: {e}")
            return processed_count


def main() -> None:
    """Main entry point for the label processor"""
    # You could load config from environment variables here
    minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'http://minio:9000')
    minio_key = os.environ.get('MINIO_KEY')
    minio_secret = os.environ.get('MINIO_SECRET')
    
    processor = LabelProcessor(minio_endpoint, minio_key, minio_secret)
    processed_count = processor.process_label_studio_results()
    
    logger.info(f"Processing complete. Processed {processed_count} files.")


if __name__ == "__main__":
    main()