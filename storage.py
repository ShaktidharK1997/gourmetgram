import s3fs
import json
import logging
import os
import boto3
import requests

logger = logging.getLogger(__name__)

class StorageManager:
    """Handles all S3/MinIO storage operations"""
    
    def __init__(self):
        self.initialize_storage()
        
    def initialize_storage(self):
        """Initialize storage connections and buckets"""
        try:
            # Initialize S3 filesystem
            self.fs = s3fs.S3FileSystem(
                key='minioadmin',
                secret='minioadmin',
                client_kwargs={
                    'endpoint_url': 'http://minio:9000' 
                }
            )
            
            # Create buckets if they don't exist
            self.BUCKET_NAME = 'production-images'
            self.TRACKING_BUCKET = 'tracking'
            self.TARGET_BUCKET = 'target-bucket'
            
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
            
            logger.info("Storage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            # Fall back to local storage in case of MinIO connection issues
            logger.warning("Falling back to local storage")
            self.fs = None
    
    def _set_bucket_public_access(self, bucket_name):
        """Set a bucket to have public read access"""
        try:
            # Create a boto3 client to interact with MinIO API
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
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
            
            # Alternative using the MinIO admin API
            try:
                # Try using MinIO's built-in API endpoint
                admin_url = "http://minio:9000/minio/admin/v3/set-bucket-policy"
                headers = {"Content-Type": "application/json"}
                data = {
                    "bucket": bucket_name,
                    "policy": "download"  # Allows public read access
                }
                
                response = requests.put(
                    admin_url, 
                    headers=headers, 
                    json=data,
                    auth=("minioadmin", "minioadmin")
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully set {bucket_name} bucket to public read access via MinIO API")
                    return True
                else:
                    logger.error(f"MinIO API error: {response.text}")
            except Exception as inner_e:
                logger.error(f"Error with MinIO API approach: {inner_e}")
            
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
                with self.fs.open(file_path, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created tracking file: {filename}")
    
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
        
        # Public URL to access image 
        return f'http://localhost:9000/{s3_path}'
    
    def append_to_tracking_file(self, file_name, entry):
        """Append an entry to a tracking file"""
        if self.fs is None:
            logger.warning("Storage not initialized, skipping tracking")
            return False
            
        file_path = f'{self.TRACKING_BUCKET}/{file_name}'
        
        try:
            # Read existing data
            if self.fs.exists(file_path):
                with self.fs.open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Append new entry
            data.append(entry)
            
            # Write back to file
            with self.fs.open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Added entry to {file_name}")
            return True
        except Exception as e:
            logger.error(f"Error appending to tracking file {file_name}: {e}")
            return False