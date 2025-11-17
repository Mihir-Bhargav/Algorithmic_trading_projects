import boto3
from botocore.config import Config

# --- Replace with your new, secure credentials ---
aws_access_key_id = 'Y3dcd3572-004b-495c-8828-6359ddd421c9'
aws_secret_access_key = 'c4qinM0qnzxzid5oZkTWiyU9OL3iBqcv'

# --- Polygon.io S3 endpoint and bucket ---
endpoint_url = 'https://files.polygon.io'
bucket_name = 'flatfiles'

# --- Example: List available option files for March 2024 ---
prefix = 'us_options_opra/trades_v1/2024/03/'

# --- Initialize session and client ---
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
s3 = session.client(
    's3',
    endpoint_url=endpoint_url,
    config=Config(signature_version='s3v4'),
)

# --- List files under the prefix ---
print("Listing available files:")
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        print(obj['Key'])

# --- Download a specific file (replace with a file you see in the list above) ---
object_key = 'us_options_opra/trades_v1/2024/03/2024-03-07.csv.gz'
local_file_name = object_key.split('/')[-1]
local_file_path = './' + local_file_name

try:
    s3.download_file(bucket_name, object_key, local_file_path)
    print(f"Downloaded {object_key} to {local_file_path}")
except Exception as e:
    print(f"Error downloading file: {e}")
