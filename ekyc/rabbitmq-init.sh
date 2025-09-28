#!/bin/bash

echo "Waiting for RabbitMQ to start..."
until rabbitmqctl status > /dev/null 2>&1; do
  sleep 2
done
echo "RabbitMQ is running!"

rabbitmq-plugins enable rabbitmq_management

sleep 5

BATCH_UPLOAD_RETRY_DELAY=${BATCH_UPLOAD_RETRY_DELAY}
BATCH_RESULT_RETRY_DELAY=${BATCH_RESULT_RETRY_DELAY}

RABBITMQ_USER=${RABBITMQ_DEFAULT_USER}
RABBITMQ_PASS=${RABBITMQ_DEFAULT_PASS}
RABBITMQ_HOST=localhost
RABBITMQ_PORT=15672

echo "Setting up RabbitMQ exchanges and queues..."

RABBITMQADMIN="rabbitmqadmin --host=${RABBITMQ_HOST} --port=${RABBITMQ_PORT} --username=${RABBITMQ_USER} --password=${RABBITMQ_PASS}"

$RABBITMQADMIN declare exchange name=x-new-request type=topic durable=true
$RABBITMQADMIN declare exchange name=x-new-request-dead-letter type=topic durable=true
$RABBITMQADMIN declare exchange name=x-request-result type=topic durable=true
$RABBITMQADMIN declare exchange name=x-request-result-dead-letter type=topic durable=true

$RABBITMQADMIN declare queue name=q-new-request durable=true \
  arguments='{"x-dead-letter-exchange":"x-new-request-dead-letter"}'

$RABBITMQADMIN declare queue name=q-new-request-dead-letter durable=true \
  arguments='{"x-dead-letter-exchange":"x-new-request","x-message-ttl":'"${BATCH_UPLOAD_RETRY_DELAY}"'}'

$RABBITMQADMIN declare queue name=q-request-result durable=true \
  arguments='{"x-dead-letter-exchange":"x-request-result-dead-letter"}'

$RABBITMQADMIN declare queue name=q-request-result-dead-letter durable=true \
  arguments='{"x-dead-letter-exchange":"x-request-result","x-message-ttl":'"${BATCH_RESULT_RETRY_DELAY}"'}'

$RABBITMQADMIN declare binding source=x-new-request destination=q-new-request destination_type=queue routing_key='#'
$RABBITMQADMIN declare binding source=x-new-request-dead-letter destination=q-new-request-dead-letter destination_type=queue routing_key='#'
$RABBITMQADMIN declare binding source=x-request-result destination=q-request-result destination_type=queue routing_key='#'
$RABBITMQADMIN declare binding source=x-request-result-dead-letter destination=q-request-result-dead-letter destination_type=queue routing_key='#'

echo "RabbitMQ setup completed!"
