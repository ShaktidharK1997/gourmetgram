#!/usr/bin/env python3
import os
import json
import random
import logging
import traceback
import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

class RandomSampler:
    """
    Randomly samples images from storage and creates Label Studio tasks for review.
    """
    
    CLASSES = [
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"
    ]
    
    def __init__(self, storage_manager, label_studio_client, logger=None):
        """
        Initialize the random sampler
        """
        self.storage = storage_manager
        self.fs = storage_manager.fs
        self.label_studio = label_studio_client
        
        # Set up logger or use provided one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("random_sampler")
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        
        # Bucket paths
        self.IMAGES_BUCKET = self.storage.BUCKET_NAME
        self.TRACKING_BUCKET = self.storage.TRACKING_BUCKET
        
        # Tracking file for random sampling
        self.TRACKING_FILE = 'random_sampling_tasks.json'
        
        # Production data tracking file
        self.PRODUCTION_DATA_FILE = 'production_data.json'
        
        # List of tracking files to check for already sampled images
        self.ALL_TRACKING_FILES = [
            'user_feedback_tasks.json',
            'low_confidence_tasks.json',
            'random_sampling_tasks.json'
        ]
    
    def get_all_images(self) -> List[Dict]:
        """Find all images in the production-images bucket"""
        all_images = []
        
        try:
            if not self.fs.exists(self.IMAGES_BUCKET):
                self.logger.error(f"Images bucket {self.IMAGES_BUCKET} does not exist")
                return []
            
            # Get class directories
            class_dirs = [d for d in self.fs.ls(self.IMAGES_BUCKET) 
                          if self.fs.isdir(d) and Path(d).name.startswith('class_')]
            
            for class_dir in class_dirs:
                class_name = Path(class_dir).name
                
                try:
                    # Get class index from directory name (class_XX)
                    class_idx = int(class_name.split('_')[1])
                    
                    # Get all images in this class directory
                    image_files = [f for f in self.fs.ls(class_dir) 
                                  if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png']]
                    
                    # Get class name from index
                    class_name = self.CLASSES[class_idx] if 0 <= class_idx < len(self.CLASSES) else "Unknown"
                    
                    # Add each image to the list
                    for img_path in image_files:
                        all_images.append({
                            'path': img_path,
                            'filename': Path(img_path).name,
                            'class_idx': class_idx,
                            'class_name': class_name
                        })
                except (IndexError, ValueError) as e:
                    self.logger.warning(f"Invalid class directory name: {class_name} - {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing directory {class_dir}: {e}")
                    continue
            
            self.logger.info(f"Found {len(all_images)} images across {len(class_dirs)} class directories")
            return all_images
            
        except Exception as e:
            self.logger.error(f"Error listing images: {e}")
            return []
    
    def get_already_sampled_images(self) -> Set[str]:
        """Get filenames of all images that have already been sampled"""
        sampled_filenames = set()
        
        for file_name in self.ALL_TRACKING_FILES:
            try:
                tasks = self.storage.read_tracking_file(file_name)
                
                # Extract filenames from image paths
                for task in tasks:
                    image_path = task.get('image_path', '')
                    if image_path:
                        filename = Path(image_path).name
                        sampled_filenames.add(filename)
            except Exception as e:
                self.logger.error(f"Error processing tracking file {file_name}: {e}")
        
        self.logger.info(f"Found {len(sampled_filenames)} already sampled image filenames")
        return sampled_filenames
    
    def get_production_confidence(self, image_path: str) -> float:
        """
        Look up the confidence score for an image from the production data
        """
        try:
            # Get filename for matching
            filename = Path(image_path).name
            
            # Read production data
            production_data = self.storage.read_tracking_file(self.PRODUCTION_DATA_FILE)
            
            # Look for matching image path or filename
            for entry in production_data:
                entry_path = entry.get('image_path', '')
                if entry_path and (entry_path == image_path or Path(entry_path).name == filename):
                    confidence = entry.get('confidence')
                    if confidence is not None:
                        self.logger.info(f"Found original confidence {confidence} for {filename}")
                        return float(confidence)
            
            # If not found, return a default high confidence
            self.logger.warning(f"No production data found for {filename}, using default confidence")
            return 0.9
            
        except Exception as e:
            self.logger.error(f"Error getting production confidence for {image_path}: {e}")
            return 0.9
    
    def sample_random_images(self, sample_count: int = 5) -> Tuple[List[Dict], List[str]]:
        """
        Sample random images and create Label Studio tasks
        """
        successful_tasks = []
        errors = []
        
        try:
            # Get all available images
            all_images = self.get_all_images()
            if not all_images:
                error_msg = "No images found in storage"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                return [], errors
            
            # Get already sampled images
            already_sampled = self.get_already_sampled_images()
            
            # Filter out already sampled images
            available_images = [img for img in all_images 
                               if img['filename'] not in already_sampled]
            
            if not available_images:
                error_msg = "No images available for sampling (all have been sampled already)"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                return [], errors
            
            # Determine how many images to sample
            sample_size = min(sample_count, len(available_images))
            sampled_images = random.sample(available_images, sample_size)
            
            self.logger.info(f"Selected {sample_size} random images for sampling")
            
            # Process each sampled image
            for img in sampled_images:
                try:
                    # Get public URL for the image
                    image_url = self.storage.get_public_url(img['path'])
                    
                    # Get the original confidence score from production data
                    confidence = self.get_production_confidence(img['path'])
                    
                    # Create task in Label Studio
                    task = self.label_studio.create_task(
                        image_url=image_url,
                        predicted_class=img['class_name'],
                        confidence=confidence,  # Use actual confidence instead of hardcoded 1.0
                        user_feedback="random_sampling"
                    )
                    
                    if not task:
                        error_msg = f"Failed to create Label Studio task for {img['filename']}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        continue
                    
                    # Create tracking entry
                    tracking_entry = {
                        "image_path": img['path'],
                        "original_prediction": img['class_name'],
                        "confidence": confidence,  # Use actual confidence from production data
                        "timestamp": datetime.datetime.now().isoformat(),
                        "task_id": task.id,
                        "status": "pending",
                        "sampling_type": "random"
                    }
                    
                    # Add to tracking file using the storage manager
                    if self.storage.append_to_tracking_file(self.TRACKING_FILE, tracking_entry):
                        successful_tasks.append(tracking_entry)
                        self.logger.info(f"Successfully sampled image: {img['filename']} with confidence {confidence}")
                    else:
                        error_msg = f"Failed to record tracking entry for {img['filename']}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing image {img['filename']}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            return successful_tasks, errors
                
        except Exception as e:
            self.logger.error(f"Error in random sampling: {e}")
            self.logger.error(traceback.format_exc())
            errors.append(f"Unexpected error: {str(e)}")
            return [], errors