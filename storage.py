import s3fs
import json
import logging
import os
import boto3
import requests
from dotenv import load_dotenv
from typing import Any, List, Dict, Optional

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StorageManager:
    """Handles all S3/MinIO storage operations"""
    
    def __init__(self):
        self.initialize_storage()
        
    def initialize_storage(self):
        """Initialize storage connections and buckets"""
        try:
            # Get environment variables
            minio_user = os.getenv('MINIO_ROOT_USER')
            minio_password = os.getenv('MINIO_ROOT_PASSWORD')
            minio_endpoint = os.getenv('MINIO_ENDPOINT')
            
            # Initialize S3 filesystem
            self.fs = s3fs.S3FileSystem(
                key=minio_user,
                secret=minio_password,
                client_kwargs={
                    'endpoint_url': minio_endpoint
                }
            )
            
            # Create buckets if they don't exist
            self.BUCKET_NAME = 'production-images'
            self.TRACKING_BUCKET = 'tracking'
            self.TARGET_BUCKET = 'target-bucket'
            self.TEST_SUITES_BUCKET = 'test-suites'
    
            self.TEST_SUITE_DIRS = ['user_feedback', 'low_confidence', 'random_sampling']
            
            # Create main bucket
            if not self.fs.exists(self.BUCKET_NAME):
                self.fs.mkdir(self.BUCKET_NAME)
                logger.info(f"Created bucket: {self.BUCKET_NAME}")
                
            # Set bucket policy to public
            self._set_bucket_public_access(self.BUCKET_NAME)
            
            # Create class subdirectories
            for i in range(10):
                class_dir = f"{self.BUCKET_NAME}/class_{i:02d}"
                if not self.fs.exists(class_dir):
                    self.fs.mkdir(class_dir)
                    logger.info(f"Created class subdirectory {class_dir}")
            
            # Create tracking bucket (this remains private)
            if not self.fs.exists(self.TRACKING_BUCKET):
                self.fs.mkdir(self.TRACKING_BUCKET)
                logger.info(f"Created bucket: {self.TRACKING_BUCKET}")
            
            # Create target bucket 
            if not self.fs.exists(self.TARGET_BUCKET):
                self.fs.mkdir(self.TARGET_BUCKET)
                logger.info(f"Created bucket: {self.TARGET_BUCKET}")
                
            # Initialize tracking files
            self._initialize_tracking_files()
            
            self._setup_test_suites_structure()
            
            logger.info("Storage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            # Fall back to local storage in case of MinIO connection issues
            logger.warning("Falling back to local storage")
            self.fs = None
    
    def _set_bucket_public_access(self, bucket_name):
        """Set a bucket to have public read access"""
        try:
            # Get environment variables
            minio_user = os.getenv('MINIO_ROOT_USER')
            minio_password = os.getenv('MINIO_ROOT_PASSWORD')
            minio_endpoint = os.getenv('MINIO_ENDPOINT')
            
            # Create a boto3 client to interact with MinIO API
            s3_client = boto3.client(
                's3',
                endpoint_url=minio_endpoint,
                aws_access_key_id=minio_user,
                aws_secret_access_key=minio_password,
                region_name='us-east-1')
          
            # Set the bucket policy to allow public read access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                    }
                ]
            }
            
            # Apply the policy to the bucket
            s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            logger.info(f"Successfully set {bucket_name} bucket to public read access")
            return True
        except Exception as e:
            logger.error(f"Error setting bucket {bucket_name} to public: {e}")
            
            return False
    
    # Public methods for JSON file handling
    def read_json_file(self, file_path: str) -> Optional[Any]:
        """ Read a JSON file from S3 with error handling """
        try:
            if not self.fs.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
                
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
    
    def write_json_file(self, file_path: str, data: Any) -> bool:
        """ Write data to a JSON file with error handling """
        try:
            with self.fs.open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return False

    def read_tracking_file(self, filename: str) -> List[Dict]:
        """ Read a tracking file from the tracking bucket """
        file_path = f"{self.TRACKING_BUCKET}/{filename}"
        result = self.read_json_file(file_path)
        return result if result is not None else []
    
    def write_tracking_file(self, filename: str, data: List[Dict]) -> bool:
        """ Write data to a tracking file """
        file_path = f"{self.TRACKING_BUCKET}/{filename}"
        return self.write_json_file(file_path, data)
    
    def append_to_tracking_file(self, filename: str, entry: Dict) -> bool:
        """ Append an entry to a tracking file """
        if self.fs is None:
            logger.warning("Storage not initialized, skipping tracking")
            return False
            
        file_path = f'{self.TRACKING_BUCKET}/{filename}'
        
        try:
            # Read existing data
            data = self.read_tracking_file(filename)
            
            # Append new entry
            data.append(entry)
            
            # Write back to file
            return self.write_tracking_file(filename, data)
        except Exception as e:
            logger.error(f"Error appending to tracking file {filename}: {e}")
            return False
            
    def _initialize_tracking_files(self):
        """Create tracking files if they don't exist"""
        tracking_files = [
            'user_feedback_tasks.json',
            'low_confidence_tasks.json',
            'random_sampling_tasks.json',
            'production_data.json'
        ]
        
        for filename in tracking_files:
            file_path = f'{self.TRACKING_BUCKET}/{filename}'
            if not self.fs.exists(file_path):
                self.write_json_file(file_path, [])
                logger.info(f"Created tracking file: {filename}")
    
    def get_task_image_paths(self, task_type: str) -> list:
        """Get paths to all images associated with a specific task type"""
        # Mapping task types to tracking files
        tracking_files = {
            'user_feedback': 'user_feedback_tasks.json',
            'low_confidence': 'low_confidence_tasks.json',
            'random_sampling': 'random_sampling_tasks.json',
            'user_corrected': 'user_corrected_labels.json'
        }
        
        if task_type not in tracking_files:
            return []
        
        tracking_file = f"{self.TRACKING_BUCKET}/{tracking_files[task_type]}"
        
        image_paths = []
        tasks = self.read_json_file(tracking_file) or []
        
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
                    class_idx = CLASSES.index(final_label)
                    
                    filename = os.path.basename(image_path)
                    
                    potential_path = f"{self.BUCKET_NAME}/class_{class_idx:02d}/{filename}"
                    if self.fs.exists(potential_path):
                        image_paths.append(potential_path)
        
        logger.info(f"Found {len(image_paths)} images for task type: {task_type}")
        return image_paths
    
    def _setup_test_suites_structure(self) -> None:
        """Create test-suites bucket and subdirectories if they don't exist"""
        # Create test-suites bucket if it doesn't exist
        if not self.fs.exists(self.TEST_SUITES_BUCKET):
            self.fs.mkdir(self.TEST_SUITES_BUCKET)
            logger.info(f"Created bucket: {self.TEST_SUITES_BUCKET}")
        
        # Create subdirectories for each test suite type
        for dir_name in self.TEST_SUITE_DIRS:
            test_suite_dir = f"{self.TEST_SUITES_BUCKET}/{dir_name}"
            if not self.fs.exists(test_suite_dir):
                self.fs.mkdir(test_suite_dir)
                logger.info(f"Created test suite directory: {test_suite_dir}")
                    
    def upload_image(self, file_data, filename, class_idx):
        """Upload an image to the appropriate class directory"""
        try:
            if self.fs is None:
                # Fall back to local storage
                local_path = f"local_storage/{filename}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(file_data)
                return local_path
            
            class_dir = f"class_{class_idx:02d}"
            s3_path = f'{self.BUCKET_NAME}/{class_dir}/{filename}'
            
            # Upload to MinIO using s3fs
            with self.fs.open(s3_path, 'wb') as s3_file:
                s3_file.write(file_data)
            
            logger.info(f"Successfully uploaded {filename} to {s3_path}")
            return s3_path
        except Exception as e:
            logger.error(f"Error uploading file to storage: {e}")
            # Return a fallback path
            return f"failed_upload/{filename}"
    
    def get_public_url(self, s3_path):
        """Get the public URL for an image"""
        if self.fs is None or s3_path.startswith("failed_upload/") or s3_path.startswith("local_storage/"):
            return f"/static/fallback_images/{os.path.basename(s3_path)}"
        
        # Get endpoint from environment
        minio_endpoint = os.getenv('MINIO_ENDPOINT')
        
        # Convert to localhost if needed for external access
        public_endpoint = minio_endpoint.replace('http://minio:9000', 'http://localhost:9000')
        
        # Public URL to access image
        return f'{public_endpoint.rstrip("/")}/{s3_path}'