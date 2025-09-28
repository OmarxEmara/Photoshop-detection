import aio_pika
from redis.asyncio import Redis
import os
import json
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv
from utils.callback import send_callback
from utils.logging_utils import get_custom_logger

load_dotenv("/home/youssef-kabodan/Code/eKYC/ekyc/.env")
logging = get_custom_logger("./mq.logs")


class AsyncRabbitMQInstance:
    def __init__(self,redis: Redis ):
        self.redis = redis
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None
        self.consuming = False
        
        self.rabbitmq_user = os.getenv("RABBITMQ_DEFAULT_USER", "guest")
        self.rabbitmq_password = os.getenv("RABBITMQ_DEFAULT_PASS", "guest")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5672))
        
        self.connection_url = (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )

    async def connect(self):
        """Establish connection to RabbitMQ"""
        if self.connection and not self.connection.is_closed:
            return
            
        try:
            self.connection = await aio_pika.connect_robust(
                self.connection_url,
                heartbeat=600,  
                blocked_connection_timeout=300, 
            )
            

            self.channel = await self.connection.channel()
            

            await self.channel.set_qos(prefetch_count=10)
            
            logging.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logging.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self):
        """Close RabbitMQ connections"""
        self.consuming = False
        
        if self.channel and not self.channel.is_closed:
            await self.channel.close()
            
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            
        logging.info("Disconnected from RabbitMQ")

    async def _ensure_connected(self):
        """Ensure we have a valid connection"""
        if not self.connection or self.connection.is_closed:
            await self.connect()

    async def declare_queue_with_dlx(self, queue_name: str, dlx_name: str) -> aio_pika.RobustQueue:
        """Declare a queue with dead letter exchange"""
        await self._ensure_connected()
        
        dlx = await self.channel.declare_exchange(
            dlx_name, 
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": dlx_name
            }
        )
        
        return queue

    async def publish_request(self, message: Dict):
        """Publish message to request queue"""
        await self._ensure_connected()
        
        queue_name = "q-new-request"
        dlx_name = "x-new-request-dead-letter"
        
        await self.declare_queue_with_dlx(queue_name, dlx_name)
        
        message_body = aio_pika.Message(
            json.dumps(message).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # Make persistent
            content_type="application/json"
        )
        
        await self.channel.default_exchange.publish(
            message_body,
            routing_key=queue_name
        )
        
        logging.info(f"Published message to {queue_name}: {message}")

    async def publish_result(self, message: Dict):
        """Publish message to result queue"""
        await self._ensure_connected()
        
        queue_name = "q-request-result"
        dlx_name = "x-request-result-dead-letter"
        
        await self.declare_queue_with_dlx(queue_name, dlx_name)
        
        message_body = aio_pika.Message(
            json.dumps(message).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json"
        )
        
        await self.channel.default_exchange.publish(
            message_body,
            routing_key=queue_name
        )
        
        logging.info(f"Published message to {queue_name}: {message}")

    async def process_message(self, message: aio_pika.IncomingMessage):
        try:
            body = json.loads(message.body.decode())
            logging.info(f"Received message: {body}")
            

            reference_id = body.get("refrence_id")  
            ocr_data = body.get("ocr_result", {})
            face_match = body.get("face_match_score", {})
            success = face_match.get("success", False)
            match = face_match.get("match", False)
            error = face_match.get("error")
            jti = body.get("jti", "")

            redis_data = {
                "ocr_data": json.dumps(ocr_data, ensure_ascii=False),
                "success": str(success),
                "match": str(match),
                "error": error or "",
            }


            if jti:
                await self.redis.hset(name=jti, mapping=redis_data)

            callback_url = await self.redis.hget(jti,"callback_url")
            await send_callback(reference_id, ocr_data, success, match, error, jti, callback_url)
            

            await message.ack()
            logging.info("Message processed successfully")
            
        except Exception as e:
            logging.exception(f"Error processing message: {e}")
            await message.reject(requeue=True)

    async def start_consuming(self, queue_name: str = "q-request-result",dlx_name= "x-request-result-dead-letter"):
        """Start consuming messages"""
        await self._ensure_connected()
        
        
        queue = await self.declare_queue_with_dlx(queue_name, dlx_name)
        
        self.consuming = True
        logging.info(f"Started consuming from {queue_name}")
        
        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if not self.consuming:
                        break
                        
                    asyncio.create_task(self.process_message(message))
                    
        except asyncio.CancelledError:
            logging.info("Consumer was cancelled")
        except Exception as e:
            logging.error(f"Error in consumer: {e}")
        finally:
            self.consuming = False

    async def stop_consuming(self):
        """Stop consuming messages"""
        self.consuming = False
        logging.info("Stopping consumer")