import logging
import datetime
import uuid
import os

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    return logger

def generate_unique_filename(original_filename):
    """Generate a unique filename with timestamp and UUID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(original_filename)
    # Add timestamp and a portion of a UUID for uniqueness
    unique_id = uuid.uuid4().hex[:8]
    return f"{name}_{timestamp}_{unique_id}{ext}"