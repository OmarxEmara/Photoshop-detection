#!/bin/bash
# filepath: /home/workstation/ashry/ekyc/message_service/start_celery_services.sh

# Start Celery worker
echo "Starting Celery worker..."
celery -A message_service.validation_service worker --loglevel=info --concurrency=10 &
CELERY_PID=$!

# Start message consumer
echo "Starting message consumer..."
python message_service/validation_service.py consumer &
CONSUMER_PID=$!

# Start Celery monitoring (optional)
echo "Starting Flower monitoring..."
celery -A message_service.validation_service flower --port=5555 &
FLOWER_PID=$!

echo "All services started. Check logs for status."

# Keep the container alive by waiting for all subprocesses
wait $CELERY_PID $CONSUMER_PID $FLOWER_PID
