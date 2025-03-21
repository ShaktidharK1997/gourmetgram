import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import datetime
import uuid
import json
from io import BytesIO

from storage import StorageManager
from label_studio_integration import LabelStudioClient
from utils import setup_logging, generate_unique_filename
from test_suite_manager import TestSuiteManager
from label_processor import LabelProcessor
from random_sampler import RandomSampler

# Set up logging
logger = setup_logging()

app = Flask(__name__)

# Initialize storage manager
storage_manager = StorageManager()

test_suite_manager = TestSuiteManager(storage_manager, logger)

label_processor = LabelProcessor(storage_manager, logger)

# Initialize Label Studio client
label_studio_client = LabelStudioClient()

# Initialize random sampler after storage_manager and label_studio_client
random_sampler = RandomSampler(storage_manager, label_studio_client, logger)

# Store temporary image data
temp_predictions = {}

# Load model and model info
try:
    model = torch.load("/app/food11.pth", map_location=torch.device('cpu'))
    with open("/app/food11_info.json", "r") as f:
        model_info = json.load(f)
    logger.info("Model and model info loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or model info: {e}")
    model = None
    model_info = None

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_data, model):
    # Handle image data directly instead of a path
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
            
            # Read file data directly
            file_data = f.read()
            
            # Make prediction using the file data directly
            predicted_class, confidence, predicted_class_idx = model_predict(file_data, model)
            
            # Upload the file to storage
            s3_path = storage_manager.upload_image(
                file_data, 
                unique_filename, 
                predicted_class_idx
            )
            
            # Get public URL
            public_url = storage_manager.get_public_url(s3_path)
            
            # Generate a prediction ID
            prediction_id = str(uuid.uuid4())
            
            # Check if confidence is low
            is_low_confidence = False
            task_id = None
            
            if model_info and confidence < model_info['thresholds']['confidence_threshold']:
                is_low_confidence = True
                
                # Automatically create a Label Studio task for low confidence predictions
                task = label_studio_client.create_task(
                    public_url,
                    predicted_class,
                    confidence,
                    "low_confidence"
                )
                
                if task:
                    task_id = task.id
                    logger.info(f"Created Label Studio task with ID: {task_id} for low confidence prediction")
                    
                    # Create tracking entry
                    tracking_entry = {
                        "image_path": s3_path,
                        "original_prediction": predicted_class,
                        "confidence": confidence,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model_version": model_info["model_info"]["version"] if model_info else "unknown",
                        "task_id": task_id,
                        "status": "pending",
                        "user_feedback": "not_available"
                    }
                    
                    # Save to tracking file
                    storage_manager.append_to_tracking_file("low_confidence_tasks.json", tracking_entry)
            
            # Store the prediction temporarily
            temp_predictions[prediction_id] = {
                'image_url': public_url,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'filename': s3_path,
                'is_low_confidence': is_low_confidence,
                'task_id': task_id  # Will be None if no task was created
            }
            
            # Store production data in tracking file
            production_data = {
                "prediction_id": prediction_id,
                "image_path": s3_path,
                "image_url": public_url,
                "prediction": predicted_class,
                "prediction_idx": int(predicted_class_idx),
                "confidence": float(confidence),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_version": model_info["model_info"]["version"] if model_info else "unknown",
                "status": "served",
                "is_low_confidence": is_low_confidence,
                "label_studio_task_id": task_id
            }
            
            # Append data to production_data.json
            storage_manager.append_to_tracking_file("production_data.json", production_data)
            
            # Return prediction to user with feedback buttons
            result_html = f'''
            <div class="prediction-result">
                <h4>Prediction Result:</h4>
                <p><span class="badge bg-info">{predicted_class}</span></p>
                <div class="d-flex justify-content-center align-items-center mt-2">
                    <small class="text-muted me-2">Was this correct?</small>
                    <button type="button" class="btn btn-link p-1 feedback-btn" data-feedback="yes" data-prediction-id="{prediction_id}" 
                            title="Correct prediction">
                        <i class="bi bi-hand-thumbs-up"></i>
                    </button>
                    <button type="button" class="btn btn-link p-1 feedback-btn" data-feedback="no" data-prediction-id="{prediction_id}"
                            title="Incorrect prediction">
                        <i class="bi bi-hand-thumbs-down"></i>
                    </button>
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
        
        # Check if a task already exists for this prediction
        existing_task_id = prediction.get('task_id')
        
        # If feedback is "no" and no task exists yet, create one
        task = None
        task_created = False
        
        if feedback == 'no' and not existing_task_id:
            # Create a task in Label Studio
            task = label_studio_client.create_task(
                image_url,
                predicted_class,
                confidence,
                "user_feedback"
            )
            
            if task:
                task_created = True
                logger.info(f"Created Label Studio task with ID: {task.id} based on user feedback")
                
                # Create tracking entry
                tracking_entry = {
                    "image_path": filename,
                    "original_prediction": predicted_class,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_version": model_info["model_info"]["version"] if model_info else "unknown",
                    "task_id": task.id,
                    "status": "pending",
                    "user_feedback": feedback
                }
                
                storage_manager.append_to_tracking_file("user_feedback_tasks.json", tracking_entry)
        
        
        # Cleanup
        del temp_predictions[prediction_id]

        # Prepare response
        task_id = task.id if task else existing_task_id
        
        response = {
            'status': 'success',
            'message': 'Thank you for your feedback!',
            'task_created': task_created,
            'task_exists': existing_task_id is not None,
            'task_id': task_id
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_test_suite', methods=['GET'])
def generate_test_suite():
    """Generate a test suite from tracking data"""
    try:
        task_type = request.args.get('task_type')
        
        valid_types = ['user_feedback', 'low_confidence', 'random_sampling', 'all']
        if task_type not in valid_types:
            return jsonify({
                "status": "error",
                "message": f"Invalid task type. Must be one of: {', '.join(valid_types)}"
            }), 400

        if task_type == 'all':
            results = test_suite_manager.create_all_test_suites()
            return jsonify({
                "status": "success",
                "message": f"Created test suites for all task types",
                "results": results
            })
        else:
            test_suite_dir = test_suite_manager.create_test_suite(task_type)
            return jsonify({
                "status": "success",
                "message": f"Created test suite for {task_type}",
                "test_suite_dir": test_suite_dir
            })
        
    except Exception as e:
        logger.error(f"Error generating test suite: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to generate test suite: {str(e)}"
        }), 500

@app.route('/process_labels', methods=['POST'])
def process_labels():
    """Process Label Studio annotation results and organize images"""
    try:
        result = label_processor.process_label_studio_results()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing labels: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to process labels: {str(e)}"
        }), 500

@app.route('/sample_random_images', methods=['POST'])
def sample_random_images():
    """Sample random images and create Label Studio tasks"""
    try:

        sample_count = 5
        if request.is_json:
            sample_count = request.json.get('sample_count', 5)
        
        successful_tasks, errors = random_sampler.sample_random_images(sample_count)

        response = {
            "status": "success" if successful_tasks else "error" if errors else "warning",
            "message": f"Created {len(successful_tasks)} random sampling tasks" if successful_tasks else "No tasks created",
            "tasks_created": len(successful_tasks),
            "requested_count": sample_count,
            "task_ids": [task.get("task_id") for task in successful_tasks if task.get("task_id")],
            "errors": errors
        }
        
        status_code = 200
        if not successful_tasks and errors:
            status_code = 500
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Error in random sampling endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to sample images: {str(e)}",
            "errors": [str(e)]
        }), 500

@app.route('/test', methods=['GET'])
def test():
    with open("/app/test_image.jpeg", "rb") as f:
        test_image_data = f.read()
    preds, probs, _ = model_predict(test_image_data, model)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)