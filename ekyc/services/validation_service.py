# import uuid 
# import os
# import traceback
# # from fastapi import UploadFile, HTTPException
# from sqlalchemy.orm import Session
# from sqlalchemy import desc
# from models.validation import RequestMetadata
# from utils.helpers import allowed_file, cleanup_files, save_file, get_dob
# from utils.callback import send_callback
# from services.matching_service import IDValidationPipeline
# from services.ocr_service import detect_and_process_id_card
# import logging
# import json
# import time
# import concurrent.futures


# validation_pipeline = IDValidationPipeline()


# UPLOAD_FOLDER = "./uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# async def validate_id_source(background_tasks, selfie: UploadFile, id_card: UploadFile, jwt_payload, db: Session, token: str):
#     if not allowed_file(selfie.filename) or not allowed_file(id_card.filename):
#         raise "Invalid file format. Only PNG and JPG/JPEG are allowed."

#     selfie_filename = f"{uuid.uuid4()}_{selfie.filename}"
#     id_filename = f"{uuid.uuid4()}_{id_card.filename}"
#     selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
#     id_path = os.path.join(UPLOAD_FOLDER, id_filename)

#     temp_files = [selfie_path, id_path]

#     try:
    

#         result = validation_pipeline.validate(selfie_path, id_path)

#         first_name, second_name, merged_name, nid, address, serial = detect_and_process_id_card(id_path)

#         result.update({
#             "FullName": merged_name,
#             "NID": nid,
#             "Address": address,
#             "Serial": serial
#         })

#         ocr_data = {
#             "fullName": merged_name,
#             "nid": nid,
#             "address": address,
#             "serial": serial,
#             "dob": get_dob(nid)
#         }

#         await send_callback(jwt_payload["reference_id"], ocr_data, result["success"], result["match"], result["error"])

#         latest = db.query(RequestMetadata).filter(
#             RequestMetadata.reference_id == jwt_payload["reference_id"]
#         ).order_by(desc(RequestMetadata.usage)).first()

#         if latest and latest.match:
#             raise HTTPException(status_code=409, detail="Already matched")

#         usage = latest.usage + 1 if latest else 1

#         db_metadata = RequestMetadata(
#             reference_id=jwt_payload["reference_id"],
#             usage=usage,
#             match=result["match"],
#             token=token,
#             selfie_path=selfie_path,
#             id_path=id_path,
#         )
#         db.add(db_metadata)
#         db.commit()

#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         background_tasks.add_task(cleanup_files, temp_files)
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

#     background_tasks.add_task(cleanup_files, temp_files)
#     return result


# executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# def kyc_callback(ch, method, properties, body, extract_fn, match_fn, publish_fn):
#     def process():
#         timer = time.time()
#         try:
#             data = json.loads(body)
#             id_card_path = data['id_card_path']
#             selfie_path = data['selfie_path']
#             print(f"[x] Processing KYC task: {id_card_path}, {selfie_path}")

#             ocr_result = extract_fn(id_card_path)
#             face_match_score = match_fn(id_card_path, selfie_path)

#             result = {
#                 'id_card_path': id_card_path,
#                 'selfie_path': selfie_path,
#                 'ocr_result': ocr_result,
#                 'face_match_score': face_match_score
#             }

#             print("[✔] Task complete. Publishing result.")
#             publish_fn("q-request-result", json.dumps(result))

#         except Exception as e:
#             logging.exception("KYC task failed.")
#             ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         else:
#             ch.basic_ack(delivery_tag=method.delivery_tag)

#         total_time = time.time() - timer
#         logging.info(f"Request to response time: {total_time:.2f}s")

#     executor.submit(process)
