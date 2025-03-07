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
    
    # Return predicted class, confidence, class index, and all probabilities
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
            
            # Create dropdown options for all classes
            classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
                "Vegetable/Fruit"]
            
            class_options = ''
            for i, class_name in enumerate(classes):
                selected = 'selected' if i == predicted_class_idx else ''
                class_options += f'<option value="{i}" {selected}>{class_name}</option>'
            
            result_html = f'''
            <div class="prediction-result">
                <h4>Prediction Result:</h4>
                <p><button type="button" class="btn btn-info">{predicted_class} ({confidence:.2f})</button></p>
                <p>Is this classification correct?</p>
                <div class="feedback-container">
                    <div class="feedback-buttons mb-3">
                        <button type="button" class="btn btn-success feedback-btn" data-feedback="yes" data-prediction-id="{prediction_id}">Yes, it's correct</button>
                        <button type="button" class="btn btn-danger" id="show-correction-{prediction_id}">No, it's incorrect</button>
                    </div>
                    <div class="correction-form" id="correction-form-{prediction_id}" style="display:none;">
                        <label for="corrected-class-{prediction_id}">Select correct class:</label>
                        <select class="form-select mb-2" id="corrected-class-{prediction_id}">
                            {class_options}
                        </select>
                        <button type="button" class="btn btn-primary submit-correction" data-prediction-id="{prediction_id}">Submit Correction</button>
                    </div>
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

@app.route('/correct', methods=['POST'])
def submit_correction():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        corrected_class_idx = int(data.get('corrected_class_idx'))
        
        if prediction_id not in temp_predictions:
            return jsonify({'status': 'error', 'message': 'Prediction ID not found'}), 404
        
        # Get prediction data
        prediction = temp_predictions[prediction_id]
        original_path = prediction['filename']
        original_class_idx = prediction['predicted_class_idx']
        
        # Get class names
        classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"]
        
        # Move the file to the corrected class directory
        new_path = storage_manager.move_image(original_path, corrected_class_idx)
        
        # Get updated public URL
        public_url = storage_manager.get_public_url(new_path)
        
        # Create a correction record
        correction_data = {
            "correction_id": str(uuid.uuid4()),
            "prediction_id": prediction_id,
            "image_path": new_path,
            "image_url": public_url,
            "original_prediction": classes[original_class_idx],
            "original_prediction_idx": original_class_idx,
            "corrected_class": classes[corrected_class_idx],
            "corrected_class_idx": corrected_class_idx,
            "original_confidence": prediction['confidence'],
            "model_version": model_info["model_info"]["version"],
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "corrected"
        }
        
        # Save correction data to user_corrected_labels.json
        storage_manager.append_to_tracking_file("user_corrected_labels.json", correction_data)
        logger.info(f"Saved user correction for prediction {prediction_id}")
        
        # Cleanup
        del temp_predictions[prediction_id]
        
        response = {
            'status': 'success',
            'message': 'Thank you for your correction! We\'ll use this to improve our model.',
            'corrected_class': classes[corrected_class_idx]
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing correction: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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

@app.route('/generate_test_suite', methods=['GET'])
def generate_test_suite():
    """Generate a test suite from user corrections"""
    try:
        # Get all corrections
        corrections = storage_manager.read_tracking_file("user_corrected_labels.json")
        
        test_suite = []
        for correction in corrections:
            test_suite.append({
                "image_path": correction["image_path"],
                "image_url": correction["image_url"],
                "correct_class": correction["corrected_class"],
                "correct_class_idx": correction["corrected_class_idx"]
            })
        
        return jsonify({
            "status": "success",
            "test_suite_size": len(test_suite),
            "test_suite": test_suite
        })
        
    except Exception as e:
        logger.error(f"Error generating test suite: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to generate test suite: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)