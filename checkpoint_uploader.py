"""A module for uploading checkpoints to GCS"""
# TODO - Implement AWS & others, dynamically detect environment.
import os
import subprocess
import time

# If running on GCS, you must login
# gcloud auth login YOUR_AUTHENTICATED_USER

# Define the source directory and destination bucket
SOURCE_DIR = "results/YOUR_MODEL_NAME"
DEST_BUCKET = "gs://YOUR_BUCKET_NAME"

exec(
    open("smol_trainer/nano_gpt/configurator.py").read()
)  # overrides from command line or config file

# Keep track of uploaded files
uploaded_files = set()


def upload_new_checkpoints():
    global uploaded_files
    # List files in the source directory
    current_files = set(os.listdir(SOURCE_DIR))

    # Determine the new files which have not been uploaded yet
    new_files = current_files - uploaded_files

    for file_name in new_files:
        file_path = os.path.join(SOURCE_DIR, file_name)
        # Execute gsutil command to copy new file to bucket
        subprocess.call(["gsutil", "cp", "-r", file_path, DEST_BUCKET])

        # Update the set of uploaded files
        uploaded_files.add(file_name)
        print(f"Uploaded {file_name} to {DEST_BUCKET}")


while True:
    upload_new_checkpoints()
    # Wait for a set duration (e.g., 10 minutes) before checking again
    time.sleep(600)  # 600 seconds is 10 minutes
