import s3fs
import json
import logging
import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StorageManager:
    """Handles all S3/MinIO storage operations with simplified design for user-corrected labels"""
    
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
            
            self.BUCKET_NAME = 'production-images'
            self.TRACKING_BUCKET = 'tracking'
            self.TEST_SUITES_BUCKET = 'test-suites'
            
            self.TEST_SUITE_DIRS = ['user_corrected']
            
            # Create main bucket
            if not self.fs.exists(self.BUCKET_NAME):
                self.fs.mkdir(self.BUCKET_NAME)
                logger.info(f"Created bucket: {self.BUCKET_NAME}")
                
            # Set bucket policy to public
            self._set_bucket_public_access(self.BUCKET_NAME)
            
            # Create class subdirectories
            for i in range(11):  # 11 food classes (0-10)
                class_dir = f"{self.BUCKET_NAME}/class_{i:02d}"
                if not self.fs.exists(class_dir):
                    self.fs.mkdir(class_dir)
                    logger.info(f"Created class subdirectory {class_dir}")
            
            # Create tracking bucket
            if not self.fs.exists(self.TRACKING_BUCKET):
                self.fs.mkdir(self.TRACKING_BUCKET)
                logger.info(f"Created bucket: {self.TRACKING_BUCKET}")
                
            # Create test suites bucket
            if not self.fs.exists(self.TEST_SUITES_BUCKET):
                self.fs.mkdir(self.TEST_SUITES_BUCKET)
                logger.info(f"Created bucket: {self.TEST_SUITES_BUCKET}")
            
            # Initialize tracking files
            self._initialize_tracking_files()
            
            # Initialize test suite directories 
            self._setup_test_suites_structure()

        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def _initialize_tracking_files(self):
        """Create tracking files if they don't exist"""
        tracking_files = [
            'production_data.json',
            'user_corrected_labels.json'
        ]
        
        for filename in tracking_files:
            file_path = f'{self.TRACKING_BUCKET}/{filename}'
            if not self.fs.exists(file_path):
                with self.fs.open(file_path, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created tracking file: {filename}")
    
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
    
    def _setup_test_suites_structure(self) -> None:
        """Create test-suites bucket and subdirectories if they don't exist"""
        try:
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
                    
        except Exception as e:
            logger.error(f"Error setting up test suites structure: {e}")
            raise
    
    def upload_image(self, file_data, filename, class_idx):
        """Upload an image to the appropriate class directory"""
        try:
            class_dir = f"class_{class_idx:02d}"
            s3_path = f'{self.BUCKET_NAME}/{class_dir}/{filename}'
            
            with self.fs.open(s3_path, 'wb') as s3_file:
                s3_file.write(file_data)
            
            return s3_path
        except Exception as e:
            logger.error(f"Error uploading file to storage: {e}")
            return None
    
    def move_image(self, original_path, new_class_idx):
        """Move an image to a different class directory based on user correction"""
        try:
            filename = original_path.split('/')[-1]
            
            new_class_dir = f"class_{new_class_idx:02d}"
            new_path = f'{self.BUCKET_NAME}/{new_class_dir}/{filename}'
            
            self.fs.copy(original_path, new_path)
            
            self.fs.rm(original_path)

            return new_path
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return original_path  # Return original path if move failed
    
    def get_public_url(self, s3_path):
        """Get the public URL for an image"""
        minio_endpoint = os.getenv('MINIO_ENDPOINT')

        public_endpoint = minio_endpoint.replace('http://minio:9000', 'http://localhost:9000')
        
        return f'{public_endpoint.rstrip("/")}/{s3_path}'
    
    def append_to_tracking_file(self, file_name, entry):
        """Append an entry to a tracking file"""
        file_path = f'{self.TRACKING_BUCKET}/{file_name}'
        
        try:
            if self.fs.exists(file_path):
                with self.fs.open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            data.append(entry)
            
            with self.fs.open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error appending to tracking file {file_name}: {e}")
            return False
    
    def read_tracking_file(self, file_name):
        """Read a tracking file"""
        file_path = f'{self.TRACKING_BUCKET}/{file_name}'
        
        try:
            if self.fs.exists(file_path):
                with self.fs.open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error reading tracking file {file_name}: {e}")
            return []