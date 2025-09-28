import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
from typing import Dict, Tuple, Optional, Union
import base64
from minio import Minio
from minio.error import S3Error
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config

config = Config()

# Replace with your MinIO server details


try:
    client = Minio(config.MINIO_ENDPOINT,
                    access_key=config.MINIO_ACCESS_KEY,
                    secret_key=config.MINIO_SECRET_KEY,
                    secure=False) 
    logging.info("MinIO connection established successfully.")
except S3Error as err:
    logging.info(f"Error connecting to MinIO: {err}")



class FaceValidator:
    """Face validation class that compares selfie with ID card face."""

    def __init__(self, model_name: str = "buffalo_l"):
        """
        Initialize face validator with InsightFace model.

        Args:
            model_name: Model name to use.
        """
        self.face_app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
        self.similarity_threshold = 0.40

    def preprocess_image(
        self, image_input: Union[str, bytes, np.ndarray]
    ) -> np.ndarray:
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
            else:
                try:
                    response = client.get_object(config.MINIO_BUCKET_NAME, image_input)
                    image_bytes = response.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except S3Error as e:
                    raise ValueError(f"Could not load image from MinIO: {e}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise ValueError("Invalid image input format")

        if img is None:
            raise ValueError("Image could not be decoded")
        return img

    def extract_face_embedding(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Extract face embedding from image.

        Args:
            image: Image as numpy array

        Returns:
            Tuple of (face embedding, face detection info) or (None, None) if no face detected
        """
        faces = self.face_app.get(image)

        if not faces:
            return None, None

        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

        face = faces[0]
        
        embedding = face.embedding

        return embedding, {
            "bbox": face.bbox.tolist(),
            "kps": face.kps.tolist() if hasattr(face, "kps") else None,
            "det_score": float(face.det_score) if hasattr(face, "det_score") else None,
        }

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two face embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity sutils between 0 and 1
        """
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def validate_face_match(
        self,
        selfie_image: Union[str, bytes, np.ndarray],
        id_image: Union[str, bytes, np.ndarray],
    ) -> Dict:
        """
        Validate if the face in selfie matches the face in ID card.

        Args:
            selfie_image: Selfie image path, bytes or numpy array
            id_image: ID card image path, bytes or numpy array

        Returns:
            Dictionary with validation results
        """
        start_time = time.time()

        selfie_img = self.preprocess_image(selfie_image)
        id_img = self.preprocess_image(id_image)

        selfie_embedding, selfie_face_info = self.extract_face_embedding(selfie_img)

        id_embedding, id_face_info = self.extract_face_embedding(id_img)

        result = {
            "success": False,
            "error": None,
            "processing_time": 0,
            "match": False,
        }

        if selfie_embedding is None:
            result["error"] = "No face detected in selfie"
            result["processing_time"] = time.time() - start_time
            return result

        if id_embedding is None:
            result["error"] = "No face detected in ID card"
            result["processing_time"] = time.time() - start_time
            return result

        similarity = self.calculate_similarity(selfie_embedding, id_embedding)
        result["match"] = similarity >= self.similarity_threshold
        result["success"] = True
        result["processing_time"] = time.time() - start_time

        return result


class IDValidationPipeline:
    """Complete ID validation pipeline that handles both ID document and face validation."""

    def __init__(self):
        """Initialize the ID validation pipeline."""
        self.face_validator = FaceValidator(model_name="buffalo_l")
        self.bucket = config.MINIO_BUCKET_NAME

    def base64_to_image(self, base64_str: str) -> np.ndarray:
        """
        Convert base64 image string to numpy array.

        Args:
            base64_str: Base64 encoded image string

        Returns:
            Image as numpy array
        """

        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def validate(
        self,
        selfie_image: Union[str, bytes, np.ndarray],
        id_image: Union[str, bytes, np.ndarray],
    ) -> Dict:
        if (
            isinstance(selfie_image, str)
            and not os.path.exists(selfie_image)
            and not selfie_image.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            selfie_image = self.base64_to_image(selfie_image)

        if (
            isinstance(id_image, str)
            and not os.path.exists(id_image)
            and not id_image.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            id_image = self.base64_to_image(id_image)

        return self.face_validator.validate_face_match(selfie_image, id_image)


def upload_image_to_minio(
    file_path: str, object_name: str, bucket_name: str = config.MINIO_BUCKET_NAME
) -> bool:
    """
    Upload an image to MinIO.

    Args:
        file_path: Local path to the image.
        object_name: Desired object name in the bucket.
        bucket_name: MinIO bucket name.

    Returns:
        True if upload was successful, False otherwise.
    """
    if not os.path.exists(file_path):
        logging.info(f"File not found: {file_path}")
        return False

    try:
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type="image/jpeg"
            if file_path.endswith(".jpg") or file_path.endswith(".jpeg")
            else "image/png",
        )
        return True
    except S3Error as e:
        logging.info(f"Failed to upload to MinIO: {e}")
        return False


if __name__ == "__main__":
    pipeline = IDValidationPipeline()

    selfie_path_local = (
        "/home/workstation/LakeHouse/sample_images/face_images/513354.jpg"
    )
    id_path_local = "/home/workstation/LakeHouse/sample_images/NID/513354.jpg"

    upload_image_to_minio(selfie_path_local, "user-uploads/selfie3.png")
    upload_image_to_minio(id_path_local, "user-uploads/id3.png")

    # selfie_path = "user-uploads/selfie.png"
    # id_path = "user-uploads/id.png"

    # result = pipeline.validate(selfie_path, id_path)
    # print(result)
