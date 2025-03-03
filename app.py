import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import s3fs
from io import BytesIO
from label_studio_sdk.client import LabelStudio
import time
import requests
import logging
import datetime
import uuid
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(
    key='minioadmin', 
    secret='minioadmin',
    client_kwargs={
        'endpoint_url': 'http://minio:9000'  # using the service name from docker-compose
    }
)
# Create tracking bucket 
TRACKING_BUCKET = 'tracking'
if not fs.exists(TRACKING_BUCKET):
    fs.mkdir(TRACKING_BUCKET)
    
#Creating tracking files
tracking_files = [
        'user_feedback_tasks.json',
        'low_confidence_tasks.json',
        'random_sampling_tasks.json'
    ]
    
for filename in tracking_files:
    file_path = f'{TRACKING_BUCKET}/{filename}'
    if not fs.exists(file_path):
        with fs.open(file_path, 'w') as f:
            json.dump([], f)
        logger.info(f"Created tracking file: {filename}")

# Function to append to tracking files
def append_to_tracking_file(file_name, entry):
    file_path = f'{TRACKING_BUCKET}/{file_name}'
    
    try:
        # Read existing data
        if fs.exists(file_path):
            with fs.open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Append new entry
        data.append(entry)
        
        # Write back to file
        with fs.open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Added entry to {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error appending to tracking file {file_name}: {e}")
        return False

# Create bucket if it doesn't exist
BUCKET_NAME = 'production-images'
if not fs.exists(BUCKET_NAME):
    fs.mkdir(BUCKET_NAME)

# Creating subdirectories inside production-images bucket 
for i in range(10):
    class_dir = f"{BUCKET_NAME}/class_{i:02d}"
    if not fs.exists(class_dir):
        fs.mkdir(class_dir)
        logger.info(f"Created class subdirectory {class_dir} inside {BUCKET_NAME} bucket")
    else:
        logger.info(f"Subdirectory already exists : {class_dir}")

# Function to wait for Label Studio to be available
def wait_for_label_studio(url, max_retries=30, retry_interval=5):
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

# Create Label Studio Client - with retry 
LABEL_STUDIO_URL = "http://label-studio:8080"
ls = None
project = None

# Use the provided token directly
API_TOKEN = "f24c69d8c610b306f43e91b19ff0aacc2c575943"
logger.info("Using provided API token for Label Studio")

# Wait for Label Studio to be available
logger.info("Waiting for Label Studio to be available...")
if wait_for_label_studio(LABEL_STUDIO_URL):
    try:
        # Use the predefined API token
        ls = LabelStudio(
            base_url=LABEL_STUDIO_URL,
            api_key=API_TOKEN
        )
        logger.info("Label Studio client initialized successfully")
        
        # Creating a project in Label Studio
        try:
            # Check if project exists first to avoid duplicates
            projects = ls.projects.list()
            project = None
            
            for p in projects:
                if p.title == "Food Classification Review":
                    project = p
                    logger.info(f"Found existing project: {project.title} (ID: {project.id})")
                    break
            
            if project is None:
                # Try to read the config file
                try:
                    with open('/app/project_setup.xml', 'r') as xml_file:
                        label_config_content = xml_file.read()
                except FileNotFoundError:
                    # If file doesn't exist, use a simple default config
                    label_config_content = '''
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
                    
                project = ls.projects.create(
                    title="Food Classification Review",
                    description="Review and correct food classification predictions",
                    label_config=label_config_content
                )
                logger.info(f"Created new project: {project.title} (ID: {project.id})")
        except Exception as e:
            logger.error(f"Error creating Label Studio project: {e}")
            project = None
    except Exception as e:
        logger.error(f"Error initializing Label Studio client: {e}")
        ls = None
else:
    logger.warning("Label Studio is not available, proceeding without it.")

# Load model - with error handling
try:
    model = torch.load("/app/food11.pth", map_location=torch.device('cpu'))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_data, model):
    # Create PIL Image from BytesIO
    img = Image.open(BytesIO(img_data)).convert('RGB')
    img = preprocess_image(img)

    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"])

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item(), predicted_class.item()

def generate_unique_filename(original_filename):
    """Generate a unique filename with timestamp and UUID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(original_filename)
    # Add timestamp and a portion of a UUID for uniqueness
    unique_id = uuid.uuid4().hex[:8]
    return f"{name}_{timestamp}_{unique_id}{ext}"

def create_label_studio_task(image_url, predicted_class, confidence, user_feedback, project_id):
    """Create a task in Label Studio with the predicted class and user feedback"""
    if not ls or not project_id:
        logger.warning("Label Studio client or project not available. Skipping task creation.")
        return None
    
    try:
        # Prepare task data according to Label Studio format
        task_data = {
            "image": image_url,
            "ml_prediction": predicted_class,
            "confidence": confidence,
            "user_feedback": user_feedback  # Add user feedback
        }
        
        # Create task with data
        task = ls.tasks.create(
            project=project_id, 
            data=task_data
        )
        
        logger.info(f"Task created successfully in Label Studio: {task.id}")
        return task
    except Exception as e:
        logger.error(f"Error creating task in Label Studio: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Store temporary image data
temp_predictions = {}

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']
            if not f:
                return '<a href="#" class="badge badge-warning">No file uploaded</a>'
                
            # Secure the filename and make it unique
            original_filename = secure_filename(f.filename)
            unique_filename = generate_unique_filename(original_filename)

            
            # Read file data into memory
            file_data = f.read()

                
            # Make prediction using the file data
            predicted_class, confidence, predicted_class_idx = model_predict(file_data, model)
            
            class_dir = f"class_{predicted_class_idx:02d}"
            s3_path = f'{BUCKET_NAME}/{class_dir}/{unique_filename}'
            
            # Upload to MinIO using s3fs
            with fs.open(s3_path, 'wb') as s3_file:
                s3_file.write(file_data)
            
            # Public URL to access image for Label Studio
            public_url = f'http://localhost:9000/{s3_path}'
            
            # Generate a prediction ID
            prediction_id = str(uuid.uuid4())
            
            # Store the prediction temporarily
            temp_predictions[prediction_id] = {
                'image_url': public_url,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'filename': s3_path
            }
            
            # Return prediction to user with feedback buttons
            result_html = f'''
            <div class="prediction-result">
                <h4>Prediction Result:</h4>
                <p><button type="button" class="btn btn-info">{predicted_class}</button></p>
                <p>Is this classification correct?</p>
                <div class="feedback-buttons">
                    <button type="button" class="btn btn-success feedback-btn" data-feedback="yes" data-prediction-id="{prediction_id}">Yes, it's correct</button>
                    <button type="button" class="btn btn-danger feedback-btn" data-feedback="no" data-prediction-id="{prediction_id}">No, it's incorrect</button>
                </div>
            </div>
            '''
            
            return result_html
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return '<a href="#" class="badge badge-danger">Error: ' + str(e) + '</a>'
    
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        feedback = data.get('feedback','not_available')
        
        if prediction_id not in temp_predictions:
            return jsonify({'status': 'error', 'message': 'Prediction ID not found'}), 404
        
        # Get prediction data
        prediction = temp_predictions[prediction_id]
        image_url = prediction['image_url']
        predicted_class = prediction['predicted_class']
        confidence = prediction['confidence']
        filename = prediction['filename']
        
        # Determine if we should create a Label Studio task
        should_create_task = False
        tracking_file = None
        
        if feedback == 'no':  # User disagrees with the classification
            should_create_task = True
            tracking_file = 'user_feedback_tasks.json'
            logger.info("Creating task because user disagreed with classification")
        elif confidence < 0.7:  # Low confidence prediction
            should_create_task = True
            tracking_file = 'low_confidence_tasks.json'
            logger.info(f"Creating task because of low confidence ({confidence:.2f})")
        
        # Create a task in Label Studio if needed
        task = None
        if should_create_task and project:
            task = create_label_studio_task(
                image_url,
                predicted_class,
                confidence,
                feedback, 
                project.id
            )
            
            if task:
                logger.info(f"Created Label Studio task with ID: {task.id}")
                task_status = "Task created successfully"
            
                if tracking_file:
                    tracking_entry = {
                            "image_path": filename,
                            "original_prediction": predicted_class,
                            "confidence": confidence,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "model_version": "v1.0",  
                            "task_id": task.id,
                            "status": "pending",
                            "user_feedback": feedback
                        }   
                    append_to_tracking_file(tracking_file, tracking_entry)
            else:
                logger.warning("Failed to create Label Studio task")
                task_status = "Failed to create task"

        del temp_predictions[prediction_id]

        response = {
            'status': 'success',
            'message': 'Thank you for your feedback!',
            'task_created': should_create_task,
            'task_id': task.id if task else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    preds, probs = model_predict("./instance/uploads/test_image.jpeg", model)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)