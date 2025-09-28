import os
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    AlGORITHM: str = "HS256"
    SECRET_KEY: str
    API_TOKEN: str
    CALLBACK_URL: str
    SQLALCHEMY_DATABASE_URL: str
    UPLOAD_FOLDER: str = "./uploads"
    RABBITMQ_DEFAULT_USER: str
    RABBITMQ_DEFAULT_PASS: str
    BATCH_UPLOAD_RETRY_DELAY: int
    BATCH_RESULT_RETRY_DELAY: int
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str
    VALIDATION_QUEUE: str = "validation_queue"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    DATABASE_URL: str = "postgresql://tazkarti_user:7fF604vGXyyP@localhost:5432/tazkarti_ekyc"
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    TEST_TOKEN: str
    RABBITMQ_HOST: str = "localhost"


    class Config:
        env_file = ".env"
