# app.py
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

# Import our custom modules
from storage import StorageManager
from utils import setup_logging, generate_unique_filename

# Set up logging
logger = setup_logging()

app = Flask(__name__)

# Initialize storage manager
storage_manager = StorageManager()

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
    model_info = {"model_info": {"version": "unknown"}}

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
        
        # Get softmax probabilities
        softmax_probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    
    # Return predicted class, confidence, class index, and all probabilities
    return classes[predicted_class.item()], torch.sigmoid(prob).item(), predicted_class.item(), softmax_probs

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
            predicted_class, confidence, predicted_class_idx, softmax_probs = model_predict(file_data, model)
            
            # Upload the file to storage in the predicted class folder
            s3_path = storage_manager.upload_image(
                file_data, 
                unique_filename, 
                predicted_class_idx
            )
            
            if not s3_path:
                return '<a href="#" class="badge badge-danger">Error uploading image</a>'
            
            # Get public URL
            public_url = storage_manager.get_public_url(s3_path)
            
            # Generate a prediction ID
            prediction_id = str(uuid.uuid4())
            
            # Store the prediction temporarily
            temp_predictions[prediction_id] = {
                'image_url': public_url,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'predicted_class_idx': predicted_class_idx,
                'softmax_probs': softmax_probs.tolist(),
                'filename': s3_path,
                'timestamp': datetime.datetime.now().isoformat()
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
                "model_version": model_info["model_info"]["version"],
                "status": "served"
            }
            
            # Append data to production_data.json
            storage_manager.append_to_tracking_file("production_data.json", production_data)
            logger.info(f"Stored production data for prediction {prediction_id}")
            
            # Upload to drift detection bucket
            drift_path = storage_manager.upload_to_drift_bucket(
                file_data,
                f"drift_{unique_filename}"
            )
            
            # Check for drift if we have enough images
            drift_result = storage_manager.check_for_drift(model_info["model_info"]["version"])
            if drift_result and drift_result.get('is_drift', False):
                logger.warning(f"DRIFT DETECTED! p-value: {drift_result['p_value']:.4f}")
            
            # Create dropdown options for all classes
            classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
                "Vegetable/Fruit"]
            
            class_options = ''
            for i, class_name in enumerate(classes):
                selected = 'selected' if i == predicted_class_idx else ''
                class_options += f'<option value="{i}" {selected}>{class_name}</option>'
            
            return '<button type="button" class="btn btn-info btn-sm">' + str(predicted_class) + '</button>' 
            
        
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
        
        response = {
            'status': 'success',
            'message': 'Thank you for your feedback!'
        }
        
        # Only additional action needed is cleanup
        if feedback == 'yes':
            # No need to move the file, it's already in the correct directory
            logger.info(f"User confirmed correct classification for {prediction_id}")
        
        # Cleanup
        del temp_predictions[prediction_id]
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/drift_status', methods=['GET'])
def drift_status():
    """Get the status of drift detection"""
    try:
        # Get latest drift detection results
        drift_results = storage_manager.read_tracking_file("drift_detection.json")
        
        # Sort by timestamp
        if drift_results:
            drift_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            latest_result = drift_results[0]
            
            return jsonify({
                "status": "success",
                "drift_detected": latest_result.get('is_drift', False),
                "p_value": latest_result.get('p_value', 1.0),
                "timestamp": latest_result.get('timestamp', ''),
                "buffer_size": latest_result.get('buffer_size', 0),
                "model_version": latest_result.get('model_version', 'unknown')
            })
        
        return jsonify({
            "status": "success",
            "message": "No drift detection results available",
            "drift_detected": False
        })
        
    except Exception as e:
        logger.error(f"Error getting drift status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get drift status: {str(e)}"
        }), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        # Read a test image from a fixed location
        with open("/app/test_image.jpeg", "rb") as f:
            test_image_data = f.read()
        preds, probs, _, _ = model_predict(test_image_data, model)
        return str(preds)
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)