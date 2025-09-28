# import os
# import io
# import logging
# import magic
# from PIL import Image
# from minio import Minio
# from minio.error import S3Error
# from dotenv import load_dotenv


# # Configure logger
# logger = logging.getLogger("minio_utils")
# logging.basicConfig(level=logging.INFO)

# # Load MinIO settings from env
# load_dotenv()
# MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
# MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
# MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# MINIO_BUCKET = os.getenv("MINIO_BUCKET", "images")
# MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# # Initialize MinIO client
# minio_client = Minio(
#     MINIO_ENDPOINT,
#     access_key=MINIO_ACCESS_KEY,
#     secret_key=MINIO_SECRET_KEY,
#     secure=MINIO_SECURE
# )

# def ensure_bucket_exists(bucket_name: str):
#     if not minio_client.bucket_exists(bucket_name):
#         minio_client.make_bucket(bucket_name)
#         logger.info(f"Created bucket: {bucket_name}")

# def validate_image(binary_data: bytes) -> str:
#     mime = magic.from_buffer(binary_data, mime=True)
#     if not mime.startswith("image/"):
#         raise ValueError("Provided binary is not a valid image")
    
#     try:
#         with Image.open(io.BytesIO(binary_data)) as img:
#             img.verify()  # Validates image
#     except Exception as e:
#         raise ValueError("Invalid image file") from e
    
#     return mime

# def upload_image_to_minio(binary_data: bytes, object_path: str, bucket_name: str = MINIO_BUCKET):
#     """
#     Validates and uploads binary image to MinIO at the given object path.
    
#     Args:
#         binary_data: Raw bytes of the image
#         object_path: Path in the bucket (e.g., 'users/1234/photo.jpg')
#         bucket_name: MinIO bucket name (default from env)

#     Raises:
#         ValueError: If the binary data is not a valid image
#         S3Error: If upload fails
#     """
#     mime_type = validate_image(binary_data)
#     ensure_bucket_exists(bucket_name)

#     try:
#         minio_client.put_object(
#             bucket_name,
#             object_path,
#             data=io.BytesIO(binary_data),
#             length=len(binary_data),
#             content_type=mime_type
#         )
#         logger.info(f"Image uploaded to {bucket_name}/{object_path}")
#     except S3Error as e:
#         logger.error(f"Failed to upload image: {e}")
#         raise


import os
import io
import logging
from minio import Minio
from minio.error import S3Error
from utils.config import Config

class MinIOManager:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        try:
            self.client = Minio(
                self.config.MINIO_ENDPOINT,
                access_key=self.config.MINIO_ACCESS_KEY,
                secret_key=self.config.MINIO_SECRET_KEY,
                secure=False
            )
            self.logger.info("MinIO connection established successfully.")
        except S3Error as err:
            self.logger.error(f"Error connecting to MinIO: {err}")
            raise
    
    def ensure_bucket_exists(self, bucket_name: str):
        """Create bucket if it doesn't exist"""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            self.logger.info(f"Created bucket: {bucket_name}")
    
    def upload_file(self, file_path: str, bucket_name: str, object_name: str = None):
        """Upload a file to MinIO bucket"""
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        self.ensure_bucket_exists(bucket_name)
        
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            self.logger.info(f"File {file_path} uploaded to {bucket_name}/{object_name}")
            return True
        except S3Error as err:
            self.logger.error(f"Error uploading file: {err}")
            return False
    
    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        """Download a file from MinIO bucket"""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            self.logger.info(f"File downloaded from {bucket_name}/{object_name} to {file_path}")
            return True
        except S3Error as err:
            self.logger.error(f"Error downloading file: {err}")
            return False
    
    def read_file_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """Read file as bytes from MinIO bucket"""
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as err:
            self.logger.error(f"Error reading file: {err}")
            raise
    
    def write_bytes(self, bucket_name: str, object_name: str, data: bytes, content_type: str = None):
        """Write bytes to MinIO bucket"""
        self.ensure_bucket_exists(bucket_name)
        
        try:
            self.client.put_object(
                bucket_name,
                object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type
            )
            self.logger.info(f"Data written to {bucket_name}/{object_name}")
            return True
        except S3Error as err:
            self.logger.error(f"Error writing data: {err}")
            return False

