import os
import tempfile
import zipfile

from constant import cfg
from services.s3 import upload_folder_to_s3


def extract_zip(zip_file_path):
    """Extracts a zip file to the specified output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the ZIP file into the temporary directory
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                # List the files in the extracted folder
                s3_prefix = zip_file_path.name.split(".")[0]  # Optional prefix for S3 (folder inside the bucket)

                file_pth = os.path.join(temp_dir, s3_prefix)
                # Initialize the S3 client with the given credentials and endpoint
                bucket_name = cfg.BUCKET_NAME  # Replace with your S3 bucket name
                folder_path = os.path.join(temp_dir, s3_prefix)
                # Upload the folder to S3
                upload_folder_to_s3(bucket_name, folder_path, s3_prefix)

                return s3_prefix
        except Exception as e:
            print(f"Error extracting {zip_file_path}: {e}")
            return None
