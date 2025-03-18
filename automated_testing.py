import requests
import time
import os
from pathlib import Path

# Configuration
url = 'http://localhost:8000/predict'
image_folder = '/home/cc/gourmetgram/food-11-images-1-class' 
image_extensions = ['.jpg', '.jpeg', '.png'] 

def main():

    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(image_folder).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    num_images = len(image_files)
    print(f"Found {num_images} images in {image_folder}")
    print(f"Starting to send requests to {url}")
    
    start_time = time.time()
    successful = 0
    
    for i, image_path in enumerate(image_files):
        try:
            with open(image_path, 'rb') as img:
                filename = os.path.basename(image_path)
                files = {'file': (filename, img, f'image/{Path(filename).suffix[1:]}') }
                response = requests.post(url, files=files)
            
            status = response.status_code
            if status == 200:
                successful += 1
                
            print(f"Request {i+1}/{num_images}: {filename} - Status {status}")
            
            # Optional: add a small delay to avoid overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Request {i+1}/{num_images}: {filename} - Error - {str(e)}")
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Successful requests: {successful}/{num_images}")

if __name__ == "__main__":
    main()