import time
import requests
import logging
from label_studio_sdk.client import LabelStudio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LabelStudioClient:
    """Handles all interactions with the Label Studio API"""
    
    def __init__(self):
        self.LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL')
        self.API_TOKEN = os.getenv('LABEL_STUDIO_USER_TOKEN')
        self.ls = None
        self.project = None

        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Label Studio client and project"""
        logger.info("Initializing Label Studio client...")
        
        # Wait for Label Studio to be available
        if self._wait_for_label_studio(self.LABEL_STUDIO_URL):
            try:
                # Initialize client with API token
                self.ls = LabelStudio(
                    base_url=self.LABEL_STUDIO_URL,
                    api_key=self.API_TOKEN
                )
                logger.info("Label Studio client initialized successfully")
                
                # Set up the project
                self._setup_project()
            except Exception as e:
                logger.error(f"Error initializing Label Studio client: {e}")
                self.ls = None
        else:
            logger.warning("Label Studio is not available, proceeding without it.")
            self.ls = None
    
    
    def _wait_for_label_studio(self, url, max_retries=10, retry_interval=5):
        """Wait for Label Studio to be available, with retries"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(f"{url}/health")
                if response.status_code == 200:
                    logger.info("Label Studio is up and running!")
                    return True
                else:
                    logger.info(f"Label Studio not ready yet. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.info(f"Label Studio not available yet: {e}")
            
            logger.info(f"Retrying in {retry_interval} seconds... (Attempt {retry_count+1}/{max_retries})")
            time.sleep(retry_interval)
            retry_count += 1
        
        logger.error("Max retries reached. Label Studio is not available.")
        return False
    
    def _setup_project(self):
        """Create or find the Label Studio project"""
        if not self.ls:
            logger.warning("Label Studio client not initialized. Skipping project setup.")
            return False
            
        try:
            # Check if project exists first to avoid duplicates
            projects = self.ls.projects.list()
            
            for p in projects:
                if p.title == "Food Classification Review":
                    self.project = p
                    logger.info(f"Found existing project: {self.project.title} (ID: {self.project.id})")
                    return True
            
            # Project not found, create a new one
            label_config_content = self._get_label_config()
                
            self.project = self.ls.projects.create(
                title="Food Classification Review",
                description="Review and correct food classification predictions",
                label_config=label_config_content
            )
            logger.info(f"Created new project: {self.project.title} (ID: {self.project.id})")
            self.connect_export_storage()
            return True
        
        except Exception as e:
            logger.error(f"Error setting up Label Studio project: {e}")
            self.project = None
            return False
    
    def connect_export_storage(self):
        """Connect the target storage bucket to the Label Studio project"""
        if not self.ls or not self.project:
            logger.warning("Label Studio client or project not available. Skipping storage connection.")
            return False
            
        try:
            # Get environment variables for MinIO
            minio_user = os.getenv('MINIO_ROOT_USER')
            minio_password = os.getenv('MINIO_ROOT_PASSWORD')
            minio_endpoint = os.getenv('MINIO_ENDPOINT')
            
            # Storage configuration for exporting annotations
            storage_config = {
                "can_delete_objects": True,
                "title": "Export Storage",
                "description": "S3 storage for exporting annotations",
                "project": self.project.id,
                "bucket": "target-bucket",  
                "aws_access_key_id": minio_user,
                "aws_secret_access_key": minio_password,
                "region_name": "us-east-1",
                "s3_endpoint": minio_endpoint
            }
            
            # Make the API call to connect storage
            response = requests.post(
                f"{self.LABEL_STUDIO_URL}/api/storages/export/s3",
                headers={
                    "Authorization": f"Token {self.API_TOKEN}",
                    "Content-Type": "application/json"
                },
                json=storage_config
            )
            
            if response.status_code in (201, 200):
                logger.info(f"Successfully connected export storage to project {self.project.id}")
                return True
            else:
                logger.error(f"Failed to connect storage: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting export storage: {e}")
            return False
    
    def _get_label_config(self):
        """Get the label config XML for the project"""
        try:
            with open('/app/project_setup.xml', 'r') as xml_file:
                return xml_file.read()
        except FileNotFoundError:
            # Use default config if file doesn't exist
            return '''
            <View>
              <Image name="image" value="$image"/>
              <Choices name="food_type" toName="image" choice="single" showInLine="true">
                <Choice value="Bread"/>
                <Choice value="Dairy product"/>
                <Choice value="Dessert"/>
                <Choice value="Egg"/>
                <Choice value="Fried food"/>
                <Choice value="Meat"/>
                <Choice value="Noodles/Pasta"/>
                <Choice value="Rice"/>
                <Choice value="Seafood"/>
                <Choice value="Soup"/>
                <Choice value="Vegetable/Fruit"/>
              </Choices>
            </View>
            '''
    
    def create_task(self, image_url, predicted_class, confidence, user_feedback):
        """Create a task in Label Studio with the predicted class and user feedback"""
        if not self.ls or not self.project:
            logger.warning("Label Studio client or project not available. Skipping task creation.")
            return None
        
        try:
            # Prepare task data according to Label Studio format
            task_data = {
                "image": image_url,
                "ml_prediction": predicted_class,
                "confidence": confidence,
                "user_feedback": user_feedback
            }
            
            # Create task with data
            task = self.ls.tasks.create(
                project=self.project.id, 
                data=task_data
            )
            
            logger.info(f"Task created successfully in Label Studio: {task.id}")
            return task
        except Exception as e:
            logger.error(f"Error creating task in Label Studio: {e}")
            return None