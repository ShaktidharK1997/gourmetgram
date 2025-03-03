# process_labels.py
import os
import json
import s3fs
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(
    key='minioadmin', 
    secret='minioadmin',
    client_kwargs={
        'endpoint_url': 'http://minio:9000'
    }
)

# Constants
TARGET_BUCKET = 'target-bucket-label-studio'
TRACKING_BUCKET = 'tracking'
LABELLED_BUCKET = 'labelled-images'
TRACKING_FILES = [
    'user_feedback_tasks.json',
    'low_confidence_tasks.json',
    'random_sampling_tasks.json'
]

# Ensure labelled bucket exists
if not fs.exists(LABELLED_BUCKET):
    fs.mkdir(LABELLED_BUCKET)
    # Create class folders
    for class_name in ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                       "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", 
                       "Vegetable/Fruit"]:
        fs.mkdir(f"{LABELLED_BUCKET}/{class_name}")

def update_tracking_status(task_id, new_label):
    """Update the status of a task in the tracking files"""
    for file_name in TRACKING_FILES:
        file_path = f'{TRACKING_BUCKET}/{file_name}'
        
        if not fs.exists(file_path):
            continue
            
        try:
            # Read file
            with fs.open(file_path, 'r') as f:
                data = json.load(f)
            
            # Update entry if found
            updated = False
            for entry in data:
                if entry.get('task_id') == task_id:
                    entry['status'] = 'labeled'
                    entry['final_label'] = new_label
                    entry['label_timestamp'] = datetime.now().isoformat()
                    updated = True
                    break
            
            # Write back if updated
            if updated:
                with fs.open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Updated task {task_id} status in {file_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating tracking file {file_name}: {e}")
    
    return False

def process_label_studio_results():
    """Process the labeled tasks in the target bucket"""
    try:
        # List all JSON files in the target bucket
        json_files = [f for f in fs.ls(TARGET_BUCKET)]
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_path in json_files:
            try:
                # Read the JSON file
                with fs.open(json_path, 'r') as f:
                    task_data = json.load(f)
                
                # Extract relevant information
                task_id = task_data.get('id')
                label_result = task_data.get('result', [])
                
                if not label_result:
                    logger.warning(f"No result found in {json_path}")
                    continue
                
                # Get the new label
                new_label = label_result[0].get('value', {}).get('choices', ['Unknown'])[0]
                
                # Get the image path
                image_url = task_data.get('task', {}).get('data', {}).get('image', '')
                if not image_url:
                    logger.warning(f"No image URL found in {json_path}")
                    continue
                
                # Extract filename from URL
                filename = image_url.split('/')[-1]
                source_path = f"production-images/{filename}"
                
                # Move to labeled bucket with correct class subfolder
                if fs.exists(source_path):
                    target_path = f"{LABELLED_BUCKET}/{new_label}/{filename}"
                    fs.copy(source_path, target_path)
                    logger.info(f"Moved {filename} to {new_label} folder")
                    
                    # Update tracking status
                    update_tracking_status(task_id, new_label)
                else:
                    logger.warning(f"Source image not found: {source_path}")
                
                # Move processed JSON to a 'processed' subfolder
                processed_dir = f"{TARGET_BUCKET}/processed"
                if not fs.exists(processed_dir):
                    fs.mkdir(processed_dir)
                    
                json_filename = json_path.split('/')[-1]
                fs.copy(json_path, f"{processed_dir}/{json_filename}")
                fs.rm(json_path)
                logger.info(f"Processed {json_filename}")
                
            except Exception as e:
                logger.error(f"Error processing {json_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error listing JSON files: {e}")

if __name__ == "__main__":
    process_label_studio_results()