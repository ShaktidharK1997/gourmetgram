import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import logging
import datetime
import uuid
from io import BytesIO

# Import our custom modules
from storage import StorageManager
from label_studio_integration import LabelStudioClient
from utils import setup_logging, generate_unique_filename

# Set up logging
logger = setup_logging()

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

# Initialize storage manager
storage_manager = StorageManager()

# Initialize Label Studio client
label_studio_client = LabelStudioClient()

# Store temporary image data
temp_predictions = {}

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
    # Check if img_data is a path or binary data
    if isinstance(img_data, str):
        # It's a file path
        img = Image.open(img_data).convert('RGB')
    else:
        # It's binary data
        img = Image.open(BytesIO(img_data)).convert('RGB')
        
    img = preprocess_image(img)

    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"])

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item(), predicted_class.item()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

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
            
            # Save file locally (like the original version)
            local_path = os.path.join(app.instance_path, 'uploads', unique_filename)
            with open(local_path, 'wb') as local_file:
                local_file.write(file_data)
            
            # Make prediction using the file data
            predicted_class, confidence, predicted_class_idx = model_predict(file_data, model)
            
            # Upload to storage
            s3_path = storage_manager.upload_image(
                file_data, 
                unique_filename, 
                predicted_class_idx
            )
            
            # Get public URL
            public_url = storage_manager.get_public_url(s3_path)
            
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
        feedback = data.get('feedback', 'not_available')
        
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
        if should_create_task:
            task = label_studio_client.create_task(
                image_url,
                predicted_class,
                confidence,
                feedback
            )
            
            if task:
                logger.info(f"Created Label Studio task with ID: {task.id}")
                
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
                    storage_manager.append_to_tracking_file(tracking_file, tracking_entry)
            else:
                logger.warning("Failed to create Label Studio task")

        # Cleanup
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
    try:
        test_image_path = "./instance/uploads/test_image.jpeg"
        preds, probs, _ = model_predict(test_image_path, model)
        return str(preds)
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)