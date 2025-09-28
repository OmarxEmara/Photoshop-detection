import cv2
import re
import pytesseract  
from ultralytics import YOLO
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import os
import sys
import numpy as np
import json
import difflib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config

config = Config()

# MinIO connection setup
try:
    client = Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=False,
    )
    print("MinIO connection established successfully.")
except S3Error as err:
    print(f"Error connecting to MinIO: {err}")

CURRENT_DIR = Path(__file__).parent
MODELS_DIR = CURRENT_DIR / "yolo_models"

# Load models globally for better performance
YOLO_ID = None
YOLO_CARD = None
YOLO_OBJECTS = None

def load_models():
    """Load YOLO models once globally for better performance"""
    global YOLO_ID, YOLO_CARD, YOLO_OBJECTS
    if YOLO_ID is None:
        YOLO_ID = YOLO(f'{MODELS_DIR}/detect_id.pt')
    if YOLO_CARD is None:
        YOLO_CARD = YOLO(f'{MODELS_DIR}/detect_id_card.pt')
    if YOLO_OBJECTS is None:
        YOLO_OBJECTS = YOLO(f'{MODELS_DIR}/detect_objects.pt')

# === Helper Functions ===

def correct_address_flexible(text, json_file_path="refrence_addresses.json"):
    """Correct address using fuzzy matching with reference addresses"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            city_area_dict = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_file_path} not found. Returning original text.")
        return text
    
    text = text.strip()
    
    # Match city first
    city_matches = difflib.get_close_matches(text, city_area_dict.keys(), n=3, cutoff=0.4)
    if not city_matches:
        possible_cities = [(city, difflib.SequenceMatcher(None, city, text).ratio()) 
                           for city in city_area_dict.keys()]
        possible_cities = [p for p in possible_cities if p[1] > 0.4]
        if possible_cities:
            city_matches = [max(possible_cities, key=lambda x: x[1])[0]]
    
    if not city_matches:
        # No good city found: return original text as both
        return f"{text} - {text}"
    
    city_correct = city_matches[0]
    possible_areas = city_area_dict[city_correct]
    
    # Match area
    area_match = difflib.get_close_matches(text, possible_areas, n=1, cutoff=0.4)
    if not area_match:
        for area in possible_areas:
            if difflib.SequenceMatcher(None, area, text).ratio() > 0.4:
                area_match = [area]
                break
    
    # Always return: city - area
    final_area = area_match[0] if area_match else text  # use original if no match
    return f"{city_correct} - {final_area}"

def split_img(image):
    """Split image horizontally into two parts"""
    mid = image.shape[0] // 2
    return image[:mid, :], image[mid:, :]

def preprocess_image(cropped_image, save=False):
    """Enhanced preprocessing function with improved interpolation"""
    # Grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # Resize to make it larger for better OCR - using INTER_CUBIC for better quality
    gray_image = cv2.resize(gray_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Threshold to binarize
    _, thresh_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh_image

def extract_text(image, bbox=None, lang='ara', config='--oem 3 --psm 6', save=False):
    """Enhanced text extraction function"""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]
    
    preprocessed_image = preprocess_image(image, save=save)
    
    # Use pytesseract to extract text from the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image, lang=lang, config=config)
    return text.strip()

def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    """Function to expand bounding box height only"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]

def crop_digits(image):
    """Enhanced digit cropping and processing"""
      # Ensure models are loaded
    results = YOLO_ID.predict(image, conf=0.1)[0]
    boxes = sorted(results.boxes.xyxy.cpu().numpy(), key=lambda b: b[0])
    
    # Filter overlapping boxes
    filtered = []
    prev = None
    for box in boxes:
        if prev is None or box[0] - prev[2] >= 1:
            filtered.append(box)
            prev = box
    
    # Process digits
    digits = []
    for box in filtered:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        digits.append(preprocess_image(crop))
    
    # Combine digits
    if not digits:
        return image, False
        
    spacing = -1
    total_w = sum(d.shape[1] for d in digits) + spacing * (len(digits) - 1)
    max_h = max(d.shape[0] for d in digits)
    final = np.ones((max_h, total_w), dtype=np.uint8) * 255
    offset = 0
    for d in digits:
        h, w = d.shape
        yoff = (max_h - h) // 2
        final[yoff:yoff + h, offset:offset + w] = d
        offset += w + spacing
    
    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    return final_bgr, len(digits) == 14

# Function to detect national ID numbers in a cropped image (legacy function)
def detect_national_id(cropped_image):
    """Legacy function - keeping for backward compatibility"""

    results = YOLO_ID(cropped_image, iou=0)
    detected_info = []
    for result in results:
        print(len(result.boxes))
        print(result.boxes)
        for box in result.boxes:
            if box.conf > 0.4:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_info.append((cls, x1))
                cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cropped_image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.imwrite("detected_id.jpg", cropped_image)

    detected_info.sort(key=lambda x: x[1])
    id_number = ''.join([str(cls) for cls, _ in detected_info])
    
    return id_number

def remove_numbers(text):
    """Function to remove numbers from a string"""
    return re.sub(r'\d+', '', text)

def decode_egyptian_id(id_number):
    """Enhanced function to decode the Egyptian ID number"""
    if len(id_number) != 14 or not id_number.isdigit():
        return {}
        
    governorates = {
        '01': 'Cairo',
        '02': 'Alexandria',
        '03': 'Port Said',
        '04': 'Suez',
        '11': 'Damietta',
        '12': 'Dakahlia',
        '13': 'Ash Sharqia',
        '14': 'Kaliobeya',
        '15': 'Kafr El-Sheikh',
        '16': 'Gharbia',
        '17': 'Monoufia',
        '18': 'El Beheira',
        '19': 'Ismailia',
        '21': 'Giza',
        '22': 'Beni Suef',
        '23': 'Fayoum',
        '24': 'El Menia',
        '25': 'Assiut',
        '26': 'Sohag',
        '27': 'Qena',
        '28': 'Aswan',
        '29': 'Luxor',
        '31': 'Red Sea',
        '32': 'New Valley',
        '33': 'Matrouh',
        '34': 'North Sinai',
        '35': 'South Sinai',
        '88': 'Foreign'
    }

    try:
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12])

        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            return {}

        gender = "Male" if gender_code % 2 != 0 else "Female"
        governorate = governorates.get(governorate_code, "Unknown")
        birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

        return {
            'birth_date': birth_date,
            'governorate': governorate,
            'gender': gender
        }
    except (ValueError, IndexError):
        return {}

def process_image(cropped_image):
    """Enhanced main processing function with improved features"""
    results = YOLO_OBJECTS(cropped_image)
    
    # Initialize info dictionary with structured format
    info = {
        'first_name': '',
        'second_name': '',
        'full_name': '',
        'nid': '',
        'address_line1': '',
        'address_line2': '',
        'address': '',  # Keep for backward compatibility
        'serial': '',
        'gender': '',
        'birth_date': '',
    }

    for result in results:
        for box in result.boxes:
            bbox = list(map(int, box.xyxy[0]))
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            if class_name == 'firstName':
                info['first_name'] = extract_text(cropped_image, bbox, lang='ara', config='--oem 3 --psm 7')
            elif class_name == 'lastName':
                info['second_name'] = extract_text(cropped_image, bbox, lang='ara', config='--oem 3 --psm 7')
            elif class_name == 'serial':
                info['serial'] = extract_text(cropped_image, bbox, lang='eng', config='--oem 3 --psm 7')
            elif class_name == 'address':
                x1, y1, x2, y2 = bbox
                address_crop = cropped_image[y1:y2, x1:x2]
                
                # Split address into two lines for better processing
                top, bottom = split_img(address_crop)
                text1 = extract_text(top, lang='ara', config='--oem 1 --psm 7')
                text2 = extract_text(bottom, lang='ara', config='--oem 3 --psm 7')
                
                info['address_line1'] = text1
                info['address_line2'] = correct_address_flexible(text2)
                info['address'] = f"{text1} {info['address_line2']}".strip()  
                
            elif class_name == 'nid':
                # Enhanced NID processing with digit cropping
                exp_bbox = expand_bbox_height(bbox, scale=1, image_shape=cropped_image.shape)
                cropped_nid = cropped_image[exp_bbox[1]:exp_bbox[3], exp_bbox[0]:exp_bbox[2]]
                
                # Try enhanced digit cropping first
                processed, valid = crop_digits(cropped_nid)
                if valid:
                    nid = extract_text(processed, lang='ara_number', config='--oem 1 --psm 7')
                else:
                    # Fallback to original method
                    custom_config = (
                        '--oem 1 '                          # Use LSTM engine
                        '--psm 7 '                          # Assume a single line of text
                    )
                    nid = extract_text(cropped_image, bbox, lang='ara_number', config=custom_config, save=True)
                
                info['nid'] = nid.replace(' ', '')

    # Generate full name and decode NID
    info['full_name'] = f"{info['first_name']} {info['second_name']}".strip()
    personal_data = decode_egyptian_id(info['nid'])
    info['gender'] = personal_data.get('gender', 'Unknown')
    info['birth_date'] = personal_data.get('birth_date', 'Unknown')
    

    return info

def load_image_from_minio(minio_client, bucket_name, image_path):
    """Load image from MinIO storage"""
    response = minio_client.get_object(bucket_name, image_path)
    image_data = np.frombuffer(response.read(), np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)


def detect_and_process_id_card(image_path, minio_client=None, bucket_name=None):
    """Enhanced function to detect and process ID card with MinIO support"""
    load_models()  # Ensure models are loaded

    if minio_client and bucket_name:
        # Read image from MinIO
        response = minio_client.get_object(bucket_name, image_path)
        image_data = response.read()

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("test.png", image)
    else:
        # Read from local file system
        image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return {}
    
    # Detect ID card in the image
    id_card_results = YOLO_CARD(image)
    
    # Process the first detected ID card
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"ID Card detected at: {x1}, {y1}, {x2}, {y2}")
            cropped_image = image[y1:y2, x1:x2]
            
            # Pass the cropped image to the processing function
            return process_image(cropped_image)
    
    print("No ID card detected in the image")
    return {}

# Legacy function wrappers for backward compatibility
def process_image_legacy(cropped_image):
    """Legacy wrapper that returns tuple format"""
    info = process_image(cropped_image)
    return (
        info['first_name'],
        info['second_name'], 
        info['full_name'],
        info['nid'],
        info['address'],
        info['serial'], 
        info['decoded_nid']['gender'],
        info['decoded_nid']['birth_date'],
    )

if __name__ == "__main__":
    
    result = detect_and_process_id_card("user-uploads/id3.png", minio_client=client, bucket_name=config.MINIO_BUCKET_NAME)
    print("Final Result:", result)