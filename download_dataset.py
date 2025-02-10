import os
import requests
from zipfile import ZipFile

# Define the Kaggle dataset URL
kaggle_url = "https://drive.google.com/file/d/1ijZ06jDHuldj6UwXdSXq0y0WBKlxYu-N/view?usp=sharing"

# Destination path
save_path = "/home/x-tazad1/penumonia_detection/dataset/chest-xray-pneumonia.zip"

# Download the file
response = requests.get(kaggle_url, stream=True)
with open(save_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# Unzip the dataset
with ZipFile(save_path, 'r') as zip_ref:
    zip_ref.extractall("/home/x-tazad1/penumonia_detection/dataset/chest-xray-pneumonia")

print("Download and extraction completed.")
