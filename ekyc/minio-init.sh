#!/bin/sh

echo "Waiting for MinIO to be ready..."

# Wait for MinIO to be available
until curl -s http://localhost:9000/minio/health/ready > /dev/null; do
  sleep 2
done

echo "MinIO is ready. Creating bucket..."

MinIO_USER=${MINIO_ROOT_USER}
MinIO_PASS=${MINIO_ROOT_PASSWORD}


echo " test ${MinIO_USER} ${MinIO_PASS} "

mc alias set local http://localhost:9000 ${MinIO_USER} ${MinIO_PASS}

mc mb local/tazkartikyc || echo "Bucket already exists"

echo "MinIO bucket 'tazkartikyc' setup completed!"
