import requests
import time
import os
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Automate testing for data drift in GourmetGram Application')
    parser.add_argument('image_folder', type=str, help='Path of folder which contains the images to be sent for testing')
    args = parser.parse_args()
    
    url = 'http://localhost:8000/predict'
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(args.image_folder).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No images found in {args.image_folder}")
        return
    
    print(f"Starting to send requests to {url}")
    print(f"Found {len(image_files)} images to process")
    
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
                print(f"Request {i+1}/{len(image_files)}: {filename} - Status {status}")
            else:
                print(f"Request {i+1}/{len(image_files)}: {filename} - Failed with status {status}: {response.text}")
                
        except Exception as e:
            print(f"Request {i+1}/{len(image_files)}: {filename} - Exception: {str(e)}")
        
        # Optional: add a small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Successful requests: {successful}/{len(image_files)}")

if __name__ == "__main__":
    main()