# storage.py
import s3fs
import json
import logging
import os
import boto3
import numpy as np
import torch
from dotenv import load_dotenv
import datetime
from alibi_detect.saving import load_detector
from PIL import Image
import io
import torchvision.transforms as transforms

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StorageManager:
    """Handles all S3/MinIO storage operations and drift detection"""
    
    def __init__(self):
        self.initialize_storage()
        self.label_buffer = []
        self.label_buffer_size = 100
        self.drift_buffer = []
        self.drift_buffer_size = int(os.getenv('BUFFER_SIZE'))
        self.drift_detector = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            self.drift_detector = load_detector('/app/detector_pt')
            logger.info("Drift detector loaded successfully")
            self.label_drift_detector = load_detector('/app/label_shift_detector_pt')
            logger.info("Label drift detector loaded successfully")
        except Exception as e:
            logger.error(f"Error loading drift detector: {e}")

        
    def initialize_storage(self):
        """Initialize storage connections and buckets"""
        try:
            # Get environment variables
            minio_user = os.getenv('MINIO_ROOT_USER')
            minio_password = os.getenv('MINIO_ROOT_PASSWORD')
            minio_endpoint = os.getenv('MINIO_ENDPOINT')
            
            self.fs = s3fs.S3FileSystem(
                key=minio_user,
                secret=minio_password,
                client_kwargs={
                    'endpoint_url': minio_endpoint
                }
            )

            self.PRODUCTION_BUCKET = 'production-images'
            self.TRACKING_BUCKET = 'tracking'
            self.DRIFT_BUCKET = 'drift-detection'  
            
            if not self.fs.exists(self.PRODUCTION_BUCKET):
                self.fs.mkdir(self.PRODUCTION_BUCKET)
                logger.info(f"Created bucket: {self.PRODUCTION_BUCKET}")
            
            if not self.fs.exists(self.TRACKING_BUCKET):
                self.fs.mkdir(self.TRACKING_BUCKET)
                logger.info(f"Created bucket: {self.TRACKING_BUCKET}")
            
            if not self.fs.exists(self.DRIFT_BUCKET):
                self.fs.mkdir(self.DRIFT_BUCKET)
                logger.info(f"Created bucket: {self.DRIFT_BUCKET}")
            
            for i in range(11):
                class_dir = f"{self.PRODUCTION_BUCKET}/class_{i:02d}"
                if not self.fs.exists(class_dir):
                    self.fs.mkdir(class_dir)
                    logger.info(f"Created class subdirectory {class_dir}")
            
            # Initialize tracking files
            self._initialize_tracking_files()
            
            logger.info("Storage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def _initialize_tracking_files(self):
        """Create tracking files if they don't exist"""
        tracking_files = [
            'production_data.json',
            'drift_detection.json',
            'drift_detection_label_shift.json'
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
            class_dir = f"class_{class_idx:02d}"
            s3_path = f'{self.PRODUCTION_BUCKET}/{class_dir}/{filename}'
            
            with self.fs.open(s3_path, 'wb') as s3_file:
                s3_file.write(file_data)
            
            return s3_path
        except Exception as e:
            logger.error(f"Error uploading file to storage: {e}")
            return None
    
    def update_label_buffer_and_check_drift(self, predicted_class_idx, model_version):
        """Add a predicted label to the buffer for label drift detection and check for drift if enough labels are collected"""
        
        self.label_buffer.append(predicted_class_idx)
        logger.info(f"Updated label buffer, size: {len(self.label_buffer)}/{self.label_buffer_size}")
        
        drift_data = None
        if len(self.label_buffer) >= self.label_buffer_size and self.label_drift_detector:
            try:
                # Prepare labels in the format expected by the detector 
                labels = np.array(self.label_buffer).reshape(-1, 1)
                
                # Run drift detection
                drift_result = self.label_drift_detector.predict(labels, return_p_val=True, return_distance=True)
                
                drift_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_version": model_version,
                    "is_drift": bool(drift_result['data']['is_drift']),
                    "p_value": float(drift_result['data']['p_val']),
                    "buffer_size": len(self.label_buffer),
                    "type": "label_drift"
                }
                
                #class distribution for analysis
                class_counts = np.bincount(np.array(self.label_buffer), minlength=11)
                class_distribution = {str(i): int(count) for i, count in enumerate(class_counts)}
                drift_data["class_distribution"] = class_distribution
                
                self.append_to_tracking_file("drift_detection_label_shift.json", drift_data)
                
                self.label_buffer = []
                logger.info("Label buffer cleared after drift check")
                
                logger.info(f"Label drift detection completed: drift detected = {drift_data['is_drift']}, p-value = {drift_data['p_value']}")
            except Exception as e:
                logger.error(f"Error checking for label drift: {e}")
        
        return drift_data
        
    def upload_to_drift_bucket_and_check_drift(self, file_data, filename, model_version):
        """Upload an image to the drift detection bucket, preprocess it for drift detection, and check for drift if enough images are in the buffer"""
        try:   
            s3_path = f'{self.DRIFT_BUCKET}/{filename}'

            img = Image.open(io.BytesIO(file_data)).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0) 
            
            img_array = img_tensor.numpy()
            
            buffer_entry = {
                'image_path': s3_path,
                'filename': filename,
                'preprocessed_image': img_array
            }
            
            if len(self.drift_buffer) == self.drift_buffer_size:
                files_to_remove = self.fs.ls(self.DRIFT_BUCKET)
                
                for file_path in files_to_remove:
                    self.fs.rm(file_path)

                self.drift_buffer.clear()
                
            with self.fs.open(s3_path, 'wb') as s3_file:
                s3_file.write(file_data)
            
            self.drift_buffer.append(buffer_entry)
            
            logger.info(f"Successfully uploaded {filename} to drift bucket, buffer size: {len(self.drift_buffer)}/{self.drift_buffer_size}")
            
            # Check for drift if we have enough images
            drift_result = None
            if len(self.drift_buffer) >= self.drift_buffer_size and self.drift_detector:
                try:
                    # Extract preprocessed images from buffer
                    preprocessed_images = np.vstack([entry['preprocessed_image'] for entry in self.drift_buffer])
                    
                    detection_result = self.drift_detector.predict(preprocessed_images, return_p_val=True, return_distance=True)
                    
                    drift_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model_version": model_version,
                        "is_drift": bool(detection_result['data']['is_drift']),
                        "p_value": float(detection_result['data']['p_val']),
                        "buffer_size": len(self.drift_buffer),
                        "buffer_filenames": [entry['filename'] for entry in self.drift_buffer],
                        "type": "feature_drift"
                    }
                    
                    self.append_to_tracking_file("drift_detection.json", drift_data)
                    drift_result = drift_data
                    logger.info(f"Drift detection completed: drift detected = {drift_data['is_drift']}, p-value = {drift_data['p_value']}")
                except Exception as e:
                    logger.error(f"Error checking for drift: {e}")
        
        except Exception as e:
            logger.error(f"Error uploading file to drift bucket: {e}")
            return None, None
    
    
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
                
            logger.info(f"Added entry to {file_name}")
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