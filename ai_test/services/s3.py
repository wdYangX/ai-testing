import os
import boto3
from botocore.exceptions import NoCredentialsError
from constant import cfg

s3_client = boto3.client(
    's3',
    endpoint_url=cfg.S3_ENDPOINT,  # Your S3-compatible service URL
    aws_access_key_id=cfg.ACCESS_KEY,  # Your access key
    aws_secret_access_key=cfg.SECRET_KEY,  # Your secret key
    region_name="us-east-1",
    config=boto3.session.Config(signature_version="s3v4")  # Using the S3v4 signature
)


def upload_folder_to_s3(bucket_name, folder_path, s3_prefix=""):
    """
    Uploads an entire folder to an S3 bucket.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        folder_path (str): The local path to the folder to upload.
        s3_prefix (str): Optional prefix in S3, which works like a folder path inside the bucket.
    """
    # Walk through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the full file path
            file_path = os.path.join(root, file)
            # Create a relative path for the file in the S3 bucket
            relative_path = os.path.relpath(file_path, folder_path)
            # Add the optional prefix (folder) to the relative path
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                # Upload the file to S3
                s3_client.upload_file(file_path, bucket_name, s3_key)
            except NoCredentialsError:
                print("Credentials not available.")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")
