import os
import gdown
import zipfile
from pathlib import Path

def download_and_extract_data():
    # Create a data directory one level above the script location
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created data directory at: {data_dir}")
    
    # Google Drive file IDs and corresponding dataset names
    datasets = {
        "baseline": "17uZU6UuwnhkIC2wMd8No2ko8ivaIFtgw",
        "indian_food_drift_case": "1l7t9FQbUVrU1N-fphtNpARBuMom78Lvn",
        "ai_images_drift_case": "1NYxBWwrWjY9RvvJB2pQQOLWkoXCP7Jdn"
    }
    
    for dataset_name, file_id in datasets.items():
        dataset_dir = data_dir / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Path for the temporary zip file
        zip_path = data_dir / f"{dataset_name}.zip"
        
        print(f"Downloading {dataset_name} dataset from Google Drive...")
        try:
            # Download the zip file
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, str(zip_path), quiet=False)
            
            # Extract the zip file to the dataset directory
            print(f"Extracting {dataset_name}.zip to {dataset_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(str(dataset_dir))
            
            # Remove the zip file after extraction
            os.remove(zip_path)
            print(f"Successfully downloaded and extracted {dataset_name} dataset to {dataset_dir}")
            
        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")

if __name__ == "__main__":
    download_and_extract_data()