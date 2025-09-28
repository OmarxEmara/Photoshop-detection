import cv2
import dlib
import logging
import os
import uuid
import numpy as np
import math
from enum import Enum

# >> config

liveness_models_dir = "livenees_models"
predictor_model_name = "shape_predictor_68_face_landmarks.dat"
predictor_path = os.path.join(liveness_models_dir, predictor_model_name)

logs_dir_path = "logs"

model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ]
)

RIGHT_THERSHOULD = 20
LEFT_THERSHOULD = -20

# Smile detection thresholds
NORMALIZED_WIDTH_THRESHOLD = 0.70
NORMALIZED_AREA_THRESHOLD = 0.10

# Dlib landmark indices
MOUTH_AREA_POINTS_IDX = [48, 50, 51, 52, 54, 57]
MOUTH_CORNERS_IDX = [48, 54]
INTER_OCULAR_POINTS_IDX = [36, 45]  # Outer corners of the eyes


# >> ENUMS


class instructions(Enum):
    instructions = ["straight", "left", "right", "smile", "blink"]


# >> setup

if not os.path.exists(logs_dir_path):
    os.makedirs(logs_dir_path)

logger_file_path = os.path.join(logs_dir_path, f"{uuid.uuid4()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(logger_file_path, mode="w"), logging.StreamHandler()],
    force=True,
)

logging.info("Loading Liveness models...")


if not os.path.exists(predictor_path):
    logging.error("Landmarks face model does not exists ")
    raise FileNotFoundError(
        f"Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    )


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

logging.info("Liveness models loaded.")


logging.info("Liveness component started successfully.")


# >> Head Orientation


def check_head_orientation(frame: np.ndarray, instruction: str) -> bool:
    if instruction not in ["straight", "left", "right"]:
        logging.error("Unknown Instruction")
        raise ValueError(
            "Unknown Instruction"
        )
    
    success = ""
    
    # >>> camera parameters 

    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[2] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))

    # >>> convert frame to gray scale and detect faces

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        logging.info("No face detected in the frame")
        return False
    elif len(faces) > 1:
        logging.info("Please ensure only one face is in the frame.")
        return False

    face = faces[0]

    # >>> detect face landmarks and get face rotation

    landmarks = predictor(gray, face)
    image_points = np.array(
        [
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),  # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
        ],
        dtype="double",
    )

    _, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    yaw_degrees = -math.degrees(
        math.atan2(
            rotation_matrix[2, 0],
            math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
        )
    )

    avg_yaw = yaw_degrees

    result = False

    if instruction == "right":
        result = avg_yaw > RIGHT_THERSHOULD
    elif instruction == "left":
        result = avg_yaw < LEFT_THERSHOULD
    else:
        result = LEFT_THERSHOULD <= avg_yaw <= RIGHT_THERSHOULD

    logging.info(f"Instruction: {instruction}, YAW: {avg_yaw:.2f}, result: {result}")

    return result


# >> Smile


def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two dlib points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        logging.info("No face detected in the frame")
        return False

    elif len(faces) > 1:
        logging.info("Please ensure only one face is in the frame.")
        return False

    final_smile_status = "No Face Detected"

    face = faces[0]
    landmarks = predictor(gray, face)

    eye_p1 = landmarks.part(INTER_OCULAR_POINTS_IDX[0])
    eye_p2 = landmarks.part(INTER_OCULAR_POINTS_IDX[1])
    inter_ocular_distance = euclidean_distance(eye_p1, eye_p2)

    if inter_ocular_distance == 0:
        inter_ocular_distance = 1e-6

    mouth_p1 = landmarks.part(MOUTH_CORNERS_IDX[0])
    mouth_p2 = landmarks.part(MOUTH_CORNERS_IDX[1])
    mouth_width = euclidean_distance(mouth_p1, mouth_p2)
    normalized_width = mouth_width / inter_ocular_distance

    area_points = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH_AREA_POINTS_IDX
    ]
    area_points_np = np.array(area_points, dtype=np.int32)
    hull = cv2.convexHull(area_points_np)
    area = cv2.contourArea(hull)
    normalized_area = area / (inter_ocular_distance**2)

    is_smiling_width = normalized_width > NORMALIZED_WIDTH_THRESHOLD
    is_smiling_area = normalized_area > NORMALIZED_AREA_THRESHOLD

    if is_smiling_width or (is_smiling_area and is_smiling_width):
        smile_status = True
    else:
        smile_status = False

    final_smile_status = smile_status

    logging.info(
        f"Smile detection: width={normalized_width:.2f}, area={normalized_area:.2f}, result={final_smile_status}"
    )

    return final_smile_status


# >> Blink


def check_blink(frame: np.ndarray) -> bool:
    def compute_EAR(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        logging.info("No face detected in the frame")
        return False

    elif len(faces) > 1:
        logging.info("Please ensure only one face is in the frame.")
        return False

    face = faces[0]
    landmarks = predictor(gray, face)

    left_eye = np.array(
        [[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)]
    )
    right_eye = np.array(
        [[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)]
    )

    left_EAR = compute_EAR(left_eye)
    right_EAR = compute_EAR(right_eye)
    avg_EAR = (left_EAR + right_EAR) / 2.0

    avg_ear = avg_EAR  # Use single frame
    logging.info(f"EAR: {avg_ear:.3f}")

    BLINK_THRESHOLD = 0.22

    logging.info(
        f"Blink detection: EAR={avg_ear:.3f}, Blink Detected: {avg_ear < BLINK_THRESHOLD}"
    )

    result = bool(avg_ear < BLINK_THRESHOLD)
    return result


# >> Main Function


def process_frame(frame: np.ndarray, test_name: str) -> bool:
    test_name = test_name.lower().strip()

    if test_name not in instructions.instructions.value:
        logging.error("Unknown Instruction")
        raise ValueError("Unknown Instruction")

    if test_name in ["straight", "left", "right"]:
        logging.info(f"Started Head Orientation Test , Instruction {test_name} ")
        return check_head_orientation(frame, test_name)
    elif test_name == "smile":
        logging.info(f"Started Smile Test")
        return detect_smile(frame)
    elif test_name == "blink":
        logging.info(f"Started Blink Test")
        return check_blink(frame)
    else:
        return False