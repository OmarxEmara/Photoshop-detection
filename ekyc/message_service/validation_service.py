import os
import sys
import json
import logging
import pika
import time
from celery import Celery
from celery.signals import worker_process_init, task_success, task_failure
from minio import Minio
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from message_service import RabbitMQPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s-%(process)d] [%(levelname)s] %(message)s"
)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.minio import MinIOManager
from services.matching_service import IDValidationPipeline
from services.ocr_service import detect_and_process_id_card

config = Config()

minio_client = None
validation_pipeline = None
rabbitmq_publisher = None

thread_local = threading.local()

app = Celery('message_service.validation_service')
app.conf.update(
    broker_url=f'pyamqp://{config.RABBITMQ_DEFAULT_USER}:{config.RABBITMQ_DEFAULT_PASS}@{os.getenv("RABBITMQ_HOST", "localhost")}:5672//',
    result_backend='rpc://',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=8,  
    task_acks_late=True,
    worker_max_tasks_per_child=50,  
    worker_concurrency=None, 
    task_default_retry_delay=10,
    task_max_retries=3,
    task_retry_backoff=True,
    include=['message_service.validation_service']
)


def get_thread_local_minio():
    """Get thread-local MinIO client"""
    if not hasattr(thread_local, 'minio_client'):
        thread_local.minio_client =MinIOManager().client
    return thread_local.minio_client

def get_thread_local_validation_pipeline():
    """Get thread-local validation pipeline"""
    if not hasattr(thread_local, 'validation_pipeline'):
        thread_local.validation_pipeline = IDValidationPipeline()
    return thread_local.validation_pipeline

def process_ocr_task(id_card_path, task_id):
    """Process OCR in a separate thread"""
    try:
        logging.info(f"[TASK {task_id}] Starting OCR processing in thread...")
        minio_client = get_thread_local_minio()
        
        ocr_result = detect_and_process_id_card(
            id_card_path,
            minio_client,
            bucket_name=config.MINIO_BUCKET_NAME
        )
        
        logging.info(f"[TASK {task_id}] OCR processing completed in thread")
        return {'type': 'ocr', 'result': ocr_result}
        
    except Exception as e:
        logging.exception(f"[TASK {task_id}] OCR processing failed in thread: {e}")
        raise

def process_face_matching_task(id_card_path, selfie_path, task_id):
    """Process face matching in a separate thread"""
    try:
        logging.info(f"[TASK {task_id}] Starting face matching in thread...")
        validation_pipeline = get_thread_local_validation_pipeline()
        
        face_match_score = validation_pipeline.validate(id_card_path, selfie_path)
        
        logging.info(f"[TASK {task_id}] Face matching completed in thread")
        return {'type': 'face_match', 'result': face_match_score}
        
    except Exception as e:
        logging.exception(f"[TASK {task_id}] Face matching failed in thread: {e}")
        raise

@worker_process_init.connect
def init_worker_process(sender=None, conf=None, **kwargs):
    """Initialize each worker process"""
    global minio_client, validation_pipeline, rabbitmq_publisher
    
    try:
        logging.info("Initializing worker process...")
        minio_client = MinIOManager().client
        logging.info("MinIO client initialized")
        
        validation_pipeline = IDValidationPipeline()
        logging.info("Validation pipeline initialized")
        
        rabbitmq_publisher = RabbitMQPublisher()
        logging.info("RabbitMQ publisher initialized")

        logging.info("Celery worker initialized successfully")
        
    except Exception as e:
        logging.exception("Celery worker initialization failed")
        raise

@task_success.connect
def task_success_handler(sender=None, task_id=None, result=None, retval=None, **kwargs):
    """Handle successful task completion"""
    global rabbitmq_publisher
    
    try:
        if rabbitmq_publisher and result:
            result['task_id'] = task_id
            result['status'] = 'success'
            result['timestamp'] = time.time()
            
            rabbitmq_publisher.publish_result(result)
            logging.info(f"Published success result for task {task_id}")
    
    except Exception as e:
        logging.exception(f"Failed to publish success result for task {task_id}: {e}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
    """Handle failed task"""
    global rabbitmq_publisher
    
    try:
        if rabbitmq_publisher:
            error_result = {
                'task_id': task_id,
                'status': 'failed',
                'error': str(exception),
                'error_details': str(einfo),
                'timestamp': time.time()
            }
            
            rabbitmq_publisher.publish_result(error_result)
            logging.error(f"Published failure result for task {task_id}")
    
    except Exception as e:
        logging.exception(f"Failed to publish failure result for task {task_id}: {e}")

@app.task(bind=True, max_retries=3, default_retry_delay=10, name='message_service.validation_service.process_kyc_task')
def process_kyc_task(self, jti, reference_id, id_card_path, selfie_path, original_message_data=None):
    """Celery task for processing KYC with parallel processing"""
    global minio_client, validation_pipeline
    
    start_time = time.time()
    task_id = self.request.id
    attempt = self.request.retries + 1
    
    try:
        logging.info(f"[TASK {task_id}] Starting parallel KYC processing (attempt {attempt}): {id_card_path}, {selfie_path}")
        
        # Initialize results
        ocr_result = None
        face_match_score = None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"KYC-{task_id}") as executor:
            # Submit both tasks
            futures = {
                executor.submit(process_ocr_task, id_card_path, task_id): 'ocr',
                executor.submit(process_face_matching_task, id_card_path, selfie_path, task_id): 'face_match'
            }
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    result = future.result()
                    if result['type'] == 'ocr':
                        ocr_result = result['result']
                        logging.info(f"[TASK {task_id}] OCR task completed")
                    elif result['type'] == 'face_match':
                        face_match_score = result['result']
                        logging.info(f"[TASK {task_id}] Face matching task completed")
                        
                except Exception as e:
                    logging.exception(f"[TASK {task_id}] {task_type} task failed: {e}")
                    raise
        
        # Prepare result
        result = {
            "reference_id": reference_id,
            "jti": jti,
            'id_card_path': id_card_path,
            'selfie_path': selfie_path,
            'ocr_result': ocr_result,
            'face_match_score': face_match_score,
            'processing_time': time.time() - start_time,
            'retry_count': self.request.retries,
            'parallel_processing': True
        }
        
        logging.info(f"[TASK {task_id}] Parallel KYC processing completed successfully in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logging.exception(f"[TASK {task_id}] Parallel KYC processing failed (attempt {attempt}): {e}")
        
        if attempt < self.max_retries + 1:
            logging.warning(f"[TASK {task_id}] Retrying in 10 seconds...")
            raise self.retry(countdown=10, exc=e)
        else:
            logging.error(f"[TASK {task_id}] Max retries exceeded, task failed permanently")
            raise

# @app.task(bind=True, max_retries=3, default_retry_delay=10, name='message_service.validation_service.process_batch_kyc_task')
# def process_batch_kyc_task(self, batch_requests):
#     """Process multiple KYC requests in parallel"""
#     task_id = self.request.id
#     start_time = time.time()
    
#     try:
#         logging.info(f"[BATCH {task_id}] Starting batch processing of {len(batch_requests)} requests")
        
#         results = []
        
#         # Process batch in parallel
#         with ThreadPoolExecutor(max_workers=min(len(batch_requests), 4), thread_name_prefix=f"Batch-{task_id}") as executor:
#             # Submit all requests
#             future_to_request = {
#                 executor.submit(
#                     process_single_kyc_request,
#                     req["refrence_id"],
#                     req['id_card_path'], 
#                     req['selfie_path'],
#                     f"{task_id}-{i}"
#                 ): req for i, req in enumerate(batch_requests)
#             }
            
#             # Collect results
#             for future in as_completed(future_to_request):
#                 request_data = future_to_request[future]
#                 try:
#                     result = future.result()
#                     result['original_message'] = request_data
#                     results.append(result)
#                 except Exception as e:
#                     logging.exception(f"[BATCH {task_id}] Request failed: {e}")
#                     results.append({
#                         'id_card_path': request_data['id_card_path'],
#                         'selfie_path': request_data['selfie_path'],
#                         'error': str(e),
#                         'original_message': request_data
#                     })
        
#         batch_result = {
#             'batch_id': task_id,
#             'total_requests': len(batch_requests),
#             'successful_requests': len([r for r in results if 'error' not in r]),
#             'failed_requests': len([r for r in results if 'error' in r]),
#             'results': results,
#             'batch_processing_time': time.time() - start_time,
#             'parallel_processing': True
#         }
        
#         logging.info(f"[BATCH {task_id}] Batch processing completed in {time.time() - start_time:.2f}s")
#         return batch_result
        
#     except Exception as e:
#         logging.exception(f"[BATCH {task_id}] Batch processing failed: {e}")
#         raise

def process_single_kyc_request(jti, reference_id, id_card_path, selfie_path, sub_task_id):
    """Process a single KYC request (used in batch processing)"""
    try:
        minio_client = get_thread_local_minio()
        validation_pipeline = get_thread_local_validation_pipeline()
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(
                detect_and_process_id_card,
                id_card_path,
                minio_client,
                config.MINIO_BUCKET_NAME
            )
            
            face_match_future = executor.submit(
                validation_pipeline.validate,
                id_card_path,
                selfie_path
            )
            ocr_result = ocr_future.result()
            face_match_score = face_match_future.result()
        
        return {
            'jti': jti,
            'reference_id': reference_id,
            'id_card_path': id_card_path,
            'selfie_path': selfie_path,
            'ocr_result': ocr_result,
            'face_match_score': face_match_score,
            'processing_time': time.time() - start_time,
            'sub_task_id': sub_task_id
        }
        
    except Exception as e:
        logging.exception(f"Single KYC request failed: {e}")
        raise

def consume_and_dispatch():
    """Consumer that receives messages and dispatches them to Celery workers"""
    try:
        logging.info("Starting consumer with parallel processing support...")
        
        credentials = pika.PlainCredentials(config.RABBITMQ_DEFAULT_USER, config.RABBITMQ_DEFAULT_PASS)
        parameters = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "localhost"),
            port=int(os.getenv("RABBITMQ_PORT", 5672)),
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        logging.info("Connected to RabbitMQ for consuming")
        
        channel.queue_declare(
            queue='q-new-request',
            durable=True,
            arguments={'x-dead-letter-exchange': 'x-new-request-dead-letter'}
        )
        
        
        def callback(ch, method, properties, body):
            try:
                logging.info(f"Received message: {method.delivery_tag}")
                
                data = json.loads(body)
                    
          
                # Process as single request
                jti = data.get('jti')
                reference_id = data.get('reference_id')
                id_card_path = data.get('id_card_path')
                selfie_path = data.get('selfie_path')
                
                if not id_card_path or not selfie_path:
                    logging.error(f"Invalid message data: {data}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    return
                
                logging.info(f"Dispatching request to Celery: {id_card_path}, {selfie_path}")
                
                task = process_kyc_task.delay(
                    jti=jti,
                    reference_id=reference_id,
                    id_card_path=id_card_path,
                    selfie_path=selfie_path,
                    original_message_data=data
                )
                
                logging.info(f"Dispatched to Celery task {task.id}")
            
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logging.exception(f"Failed to dispatch message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        channel.basic_qos(prefetch_count=2)
        channel.basic_consume(queue='q-new-request', on_message_callback=callback)
        logging.info("Started consuming messages for parallel Celery dispatch")
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            logging.info("Stopping consumer...")
            channel.stop_consuming()
            connection.close()
            
    except Exception as e:
        logging.exception("Consumer failed to start")
        raise

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'consumer':
        consume_and_dispatch()
    else:
        app.worker_main()

__all__ = ['app', 'process_kyc_task']