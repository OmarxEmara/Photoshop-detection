import os
import io
import logging
import magic
from PIL import Image
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import asyncio


# Configure logger
logger = logging.getLogger("minio_utils")
logging.basicConfig(level=logging.INFO)

# Load MinIO settings from env
load_dotenv()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9111")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "9fFVZ9ggu80t")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "tazkartikyc")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false")

# Initialize MinIO client
minio_client = Minio(
 MINIO_ENDPOINT, access_key="minioadmin", secret_key="9fFVZ9ggu80t", secure=False
)


def ensure_bucket_exists(bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(f"Created bucket: {bucket_name}")


def validate_image(binary_data: bytes) -> str:
    mime = magic.from_buffer(binary_data, mime=True)
    print(mime)
    if not mime.startswith("image/"):
        raise ValueError("Provided binary is not a valid image")

    try:
        with Image.open(io.BytesIO(binary_data)) as img:
            img.verify()  # Validates image
    except Exception as e:
        raise ValueError("Invalid image file") from e

    return mime


def upload_image_to_minio(
    binary_data: bytes, object_path: str, bucket_name: str = MINIO_BUCKET
):
    """
    Validates and uploads binary image to MinIO at the given object path.

    Args:
        binary_data: Raw bytes of the image
        object_path: Path in the bucket (e.g., 'users/1234/photo.jpg')
        bucket_name: MinIO bucket name (default from env)

    Raises:
        ValueError: If the binary data is not a valid image
        S3Error: If upload fails
    """
    mime_type = validate_image(binary_data)
    ensure_bucket_exists(bucket_name)

    try:
        minio_client.put_object(
            bucket_name,
            object_path,
            data=io.BytesIO(binary_data),
            length=len(binary_data),
            content_type=mime_type,
        )
        logger.info(f"Image uploaded to {bucket_name}/{object_path}")
    except S3Error as e:
        logger.error(f"Failed to upload image: {e}")
        raise


async def upload_image_to_minio_async(
    binary_data: bytes, object_path: str, bucket_name: str = MINIO_BUCKET
):
    """
    Async wrapper to validate and upload image to MinIO using a thread.
    """
    return await asyncio.to_thread(upload_image_to_minio, binary_data, object_path, bucket_name)


if __name__ == "__main__":
    # Example usage
    print(ensure_bucket_exists(MINIO_BUCKET))
    with open("ekyc/tests/ID-2.png", "rb") as f:
        bindata = f.read()
        print(type(bindata))
        upload_image_to_minio(bindata, "test/test_image.png")
        print("Image uploaded successfully.")
