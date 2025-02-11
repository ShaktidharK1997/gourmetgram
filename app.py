import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time 

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

model = torch.load("food11.pth", map_location=torch.device('cpu') )

# Define metrics
REQUEST_COUNT = Counter('flask_app_request_count', 'Total number of requests received')
INFERENCE_DURATION = Histogram('flask_app_inference_duration_seconds', 'Prediction inference time')
PREDICTION_PROBABILITY = Gauge('flask_app_prediction_probability', 'Probability of the predicted class')

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_image(img)

    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
	"Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
	"Vegetable/Fruit"])

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item()

@app.route('/', methods=['GET'])
def index():
    REQUEST_COUNT.inc()
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    REQUEST_COUNT.inc()
    preds = None
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        start_time = time.time()
        preds, probs = model_predict("./instance/uploads/" + secure_filename(f.filename), model)
        INFERENCE_DURATION.observe(time.time() - start_time)
        PREDICTION_PROBABILITY.set(probs)
        return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>' 
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    REQUEST_COUNT.inc()
    start_time = time.time()
    preds, probs = model_predict("./instance/uploads/test_image.jpeg", model)
    INFERENCE_DURATION.observe(time.time() - start_time)
    PREDICTION_PROBABILITY.set(probs)
    return str(preds)

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
