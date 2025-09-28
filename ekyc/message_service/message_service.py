from dataclasses import dataclass, field
import os
import pika
import time
import json
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import signal
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

config = Config()

class RMQService:
    def __init__(self, num_workers=None):
        self.user = config.RABBITMQ_DEFAULT_USER
        self.password = config.RABBITMQ_DEFAULT_PASS
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", 5672))
        
        # Use CPU count if not specified
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        
        self.connection = None
        self.channel = None
        self.workers = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.should_stop = Value('i', 0)
        
        logging.info(f"Initialized with {self.num_workers} worker processes")

    def connect(self):
        """Connect to RabbitMQ"""
        retries = 5
        for i in range(retries):
            try:
                credentials = pika.PlainCredentials(self.user, self.password)
                parameters = pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                self.channel.basic_qos(prefetch_count=self.num_workers)
                logging.info("Connected to RabbitMQ.")
                return
            except pika.exceptions.AMQPConnectionError as e:
                logging.warning(f"Connection failed ({i+1}/{retries}): {e}")
                time.sleep(5)
        raise ConnectionError("Failed to connect to RabbitMQ after retries.")

    def close(self):
        """Close connections and stop workers"""
        # Signal workers to stop
        self.should_stop.value = 1
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        
        # Close RabbitMQ connection
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        
        logging.info("All workers stopped and connection closed.")

    def start_workers(self, worker_function):
        """Start worker processes"""
        for i in range(self.num_workers):
            worker = Process(
                target=self._worker_process,
                args=(i, worker_function, self.task_queue, self.result_queue, self.should_stop)
            )
            worker.start()
            self.workers.append(worker)
            logging.info(f"Started worker process {i}")

    @staticmethod
    def _worker_process(worker_id, worker_function, task_queue, result_queue, should_stop):
        """Worker process function"""
        logging.info(f"Worker {worker_id} started")
        
        while not should_stop.value:
            try:
                # Get task with timeout
                task_data = task_queue.get(timeout=1)
                if task_data is None:  # Poison pill
                    break
                
                start_time = time.time()
                logging.info(f"Worker {worker_id} processing task")
                
                # Process the task
                result = worker_function(task_data)
                
                # Put result back
                result_queue.put({
                    'worker_id': worker_id,
                    'result': result,
                    'task_data': task_data,
                    'processing_time': time.time() - start_time
                })
                
                logging.info(f"Worker {worker_id} completed task in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                if not should_stop.value:
                    logging.exception(f"Worker {worker_id} error: {e}")
                    # Put error result
                    result_queue.put({
                        'worker_id': worker_id,
                        'error': str(e),
                        'task_data': task_data if 'task_data' in locals() else None
                    })
        
        logging.info(f"Worker {worker_id} stopped")

    def consume_and_process(self, queue_name: str, worker_function):
        """Main consumer loop"""
        if not self.connection or self.connection.is_closed:
            self.connect()

        # Declare queue
        self.channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments={'x-dead-letter-exchange': 'x-new-request-dead-letter'}
        )

        # Start worker processes
        self.start_workers(worker_function)

        # Start result handler
        result_handler = Process(target=self._result_handler, args=(self.result_queue, self.should_stop))
        result_handler.start()

        # Track pending acknowledgments
        pending_acks = {}

        def on_message(ch, method, properties, body):
            try:
                # Parse message
                data = json.loads(body)
                task_data = {
                    'data': data,
                    'delivery_tag': method.delivery_tag,
                    'message_id': properties.message_id or f"msg_{method.delivery_tag}"
                }
                
                # Add to task queue
                self.task_queue.put(task_data)
                
                # Track for acknowledgment
                pending_acks[method.delivery_tag] = {
                    'channel': ch,
                    'method': method,
                    'timestamp': time.time()
                }
                
                logging.info(f"Queued message {method.delivery_tag} for processing")

            except Exception as e:
                logging.exception(f"Failed to queue message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        # Handle acknowledgments from result queue
        def handle_results():
            while not self.should_stop.value:
                try:
                    result = self.result_queue.get(timeout=1)
                    delivery_tag = result['task_data']['delivery_tag']
                    
                    if delivery_tag in pending_acks:
                        ack_info = pending_acks.pop(delivery_tag)
                        
                        if 'error' in result:
                            # Reject message on error
                            ack_info['channel'].basic_nack(
                                delivery_tag=delivery_tag,
                                requeue=False
                            )
                            logging.error(f"Message {delivery_tag} rejected due to processing error")
                        else:
                            # Acknowledge successful processing
                            ack_info['channel'].basic_ack(delivery_tag=delivery_tag)
                            logging.info(f"Message {delivery_tag} acknowledged")
                
                except Exception as e:
                    if not self.should_stop.value:
                        logging.exception(f"Result handling error: {e}")

        # Start result acknowledgment handler
        import threading
        ack_handler = threading.Thread(target=handle_results, daemon=True)
        ack_handler.start()

        # Start consuming
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message,
            auto_ack=False
        )

        logging.info(f"Started consuming from {queue_name} with {self.num_workers} workers")
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logging.info("Stopping consumer...")
            self.channel.stop_consuming()
        finally:
            result_handler.terminate()

    def _result_handler(self, result_queue, should_stop):
        """Handle publishing results"""
        # Create separate connection for publishing
        try:
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials
            )
            pub_connection = pika.BlockingConnection(parameters)
            pub_channel = pub_connection.channel()
            
            pub_channel.queue_declare(
                queue="q-request-result",
                durable=True,
                arguments={'x-dead-letter-exchange': 'x-request-result-dead-letter'}
            )
            
            while not should_stop.value:
                try:
                    result = result_queue.get(timeout=1)
                    
                    if 'error' not in result:
                        # Publish successful result
                        result_data = result['result']
                        body = json.dumps(result_data)
                        
                        pub_channel.basic_publish(
                            exchange="",
                            routing_key="q-request-result",
                            body=body,
                            properties=pika.BasicProperties(delivery_mode=2)
                        )
                        
                        logging.info(f"Published result from worker {result['worker_id']}")
                
                except Exception as e:
                    if not should_stop.value:
                        logging.exception(f"Result publishing error: {e}")
            
            pub_connection.close()
            
        except Exception as e:
            logging.exception(f"Result handler error: {e}")

    def publish(self, queue_name: str, message: any, arguments: dict = None):
        """Simple publish method"""
        if not self.connection or self.connection.is_closed:
            self.connect()

        queue_args = arguments or {}
        self.channel.queue_declare(queue=queue_name, durable=True, arguments=queue_args)
        
        body = message if isinstance(message, str) else json.dumps(message)
        
        self.channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        logging.info(f"Published message to {queue_name}")


import pika
import logging
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from utils.config import Config


config = Config()


class RabbitMQPublisher:
    def __init__(self):
        try:
            credentials = pika.PlainCredentials(config.RABBITMQ_DEFAULT_USER, config.RABBITMQ_DEFAULT_PASS)
            parameters = pika.ConnectionParameters(
                host=os.getenv("RABBITMQ_HOST", "localhost"),
                port=int(os.getenv("RABBITMQ_PORT", 5672)),
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare result queue
            self.channel.queue_declare(
                queue='q-request-result',
                durable=True,
                arguments={'x-dead-letter-exchange': 'x-request-result-dead-letter'}
            )
            logging.info("RabbitMQ Publisher initialized successfully")
            
        except Exception as e:
            logging.exception("Failed to initialize RabbitMQ Publisher")
            raise
    
    def publish_result(self, result):
        try:
            body = json.dumps(result, default=str)
            self.channel.basic_publish(
                exchange='',
                routing_key='q-request-result',
                body=body,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            logging.info(f"Published result to q-request-result: {result.get('task_id', 'unknown')}")
        except Exception as e:
            logging.exception("Failed to publish result")
            raise
    
    def close(self):
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
        except Exception as e:
            logging.exception("Error closing RabbitMQ connection")