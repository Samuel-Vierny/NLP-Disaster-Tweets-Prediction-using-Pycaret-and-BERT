import os
import kaggle
import zipfile

DATA_DIR = "data"
COMPETITION = "nlp-getting-started"

# Ensure the directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download the dataset
kaggle.api.competition_download_files(COMPETITION, path=DATA_DIR)

# Unzip the downloaded file
zip_path = os.path.join(DATA_DIR, f"{COMPETITION}.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)  # Delete the zip file after extraction

print("Download and extraction complete.")
