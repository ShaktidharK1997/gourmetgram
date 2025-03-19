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

logger = setup_logging()

app = Flask(__name__)

storage_manager = StorageManager()

temp_predictions = {}

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

    img = Image.open(BytesIO(img_data)).convert('RGB')
    img = preprocess_image(img)
    
    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"])

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
        
        softmax_probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item(), predicted_class.item(), softmax_probs

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            if not f:
                return '<a href="#" class="badge badge-warning">No file uploaded</a>'
                
            original_filename = secure_filename(f.filename)
            unique_filename = generate_unique_filename(original_filename)
            
            file_data = f.read()
            
            predicted_class, confidence, predicted_class_idx, softmax_probs = model_predict(file_data, model)
            
            s3_path = storage_manager.upload_image(
                file_data, 
                unique_filename, 
                predicted_class_idx
            )
            
            if not s3_path:
                return '<a href="#" class="badge badge-danger">Error uploading image</a>'
            
            public_url = storage_manager.get_public_url(s3_path)
            
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
            
            # Upload for feature defect detection
            storage_manager.upload_to_drift_bucket_and_check_drift(
                file_data,
                f"drift_{unique_filename}",
                model_info["model_info"]["version"]
            )

            # Update for label drift detection
            storage_manager.update_label_buffer_and_check_drift(
                predicted_class_idx,
                model_info["model_info"]["version"]
                )
            
            return '<button type="button" class="btn btn-info btn-sm">' + str(predicted_class) + '</button>' 
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return '<a href="#" class="badge badge-danger">Error: ' + str(e) + '</a>'
    
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/drift_status', methods=['GET'])
def drift_status():
    """Get the status of drift detection"""
    
    # Get both feature and label drift results
    feature_drift_results = storage_manager.read_tracking_file("drift_detection.json")
    label_drift_results = storage_manager.read_tracking_file("drift_detection_label_shift.json")
    
    # Combine results
    all_drift_results = []
    if feature_drift_results:
        all_drift_results.extend(feature_drift_results)
    if label_drift_results:
        all_drift_results.extend(label_drift_results)
    
    if all_drift_results:
        # Sort by timestamp (newest first)
        all_drift_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            "status": "success",
            "drift_results": all_drift_results
        })
    else:
        return jsonify({
            "status": "fail",
            "message": "No drift results currently"
        })

@app.route('/admin', methods=['GET'])
def admin_dashboard():
    """Admin dashboard to view model drift results"""
    return render_template('admin.html')

@app.route('/test', methods=['GET'])
def test():
    try:
        with open("/app/test_image.jpeg", "rb") as f:
            test_image_data = f.read()
        preds, probs, _, _ = model_predict(test_image_data, model)
        return str(preds)
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)