import uuid
import json
import asyncio
from typing import Annotated
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.security import HTTPBearer
import time
import logging
from jose import jwt
from datetime import datetime, timedelta, UTC
import sqlalchemy
from models.validation import Token
from utils.config import Config
from utils.middlewares import LoggingMiddleware
import utils.minio_utils as minio_utils
from utils.logging_utils import get_custom_logger
from utils.callback import send_callback_raw
from dependencies.auth import verify_jwt, verify_token, verify_jwt_socket
from db.database import get_db, engine, Base
from schemas.validation_models import (
    CreateSessionRequest,
    CreateSessionResponse,
    HealthResponse,
    TokenRequest,
    TokenResponse
)
from fastapi.security import OAuth2PasswordBearer
import grpc.aio as grpc
import proto.liveness_pb2 as liveness_pb2
import proto.liveness_pb2_grpc as liveness_pb2_grpc
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
from appcontext import ctx


config = Config()
UPLOAD_FOLDER = config.UPLOAD_FOLDER
LOG_FILE = "./logs/ekyc.logs"
SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.AlGORITHM
CALLBACK_URL = config.CALLBACK_URL
VALIDATION_QUEUE = config.VALIDATION_QUEUE
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
security = HTTPBearer()

Base.metadata.create_all(bind=engine)
logger = get_custom_logger(LOG_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize resources"""
    try:
        await ctx.initialize()
        consumer_task = asyncio.create_task(ctx.rabbitmq.start_consuming())
        app.state.consumer_task = consumer_task
        logger.info("RabbitMQ consumer started")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise e
    yield
    logging.info("Shutting down Context")
    if hasattr(app.state, 'consumer_task'):
        await ctx.rabbitmq.stop_consuming()
    await ctx.cleanup()
    logger.info("Graceful shutdown completed succesfully")


app = FastAPI(
    title="ID Validation API",
    description="API for validating IDs using face matching and document verification",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_image_via_grpc(frame_data: bytes, instruction: str):
    channel = grpc.insecure_channel("liveness:50052")
    stub = liveness_pb2_grpc.LivenessServiceStub(channel)

    response = await stub.DetectImage(
        liveness_pb2.LivenessRequest(image_data=frame_data, instruction=instruction)
    )
    return response.match


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint"""
    return {"status": "ok", "timestamp": int(time.time())}

@app.post("/api/v1/create_session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest, _=Depends(verify_token), db=Depends(get_db)):
    """Endpoint to generate a JWT token for a user"""
    expire = datetime.now(UTC) + timedelta(minutes=15)
    jti = str(uuid.uuid4())
    
    callback_url = request.callback_url if request.callback_url else CALLBACK_URL
    await ctx.redis.hset(
        jti,
        mapping={
            "reference_id": request.reference_id,
            "liveness_status": 0,
            "matching_status": 0,
            "submitted_status": 0,
            "expire": expire.timestamp(),
            "callback_url": callback_url,
        },
    )
    await ctx.redis.expire(jti, 600)
    return CreateSessionResponse(session_id=jti,expire_timestamp=int(expire.timestamp()))

@app.post("/api/v1/token",response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    jti = request.session_id
    try:
        if not await ctx.redis.exists(jti):
            raise HTTPException(
            status_code=401, detail="Invalid or expired token"
        )
        redis_data = await ctx.redis.hgetall(jti)
        to_encode = dict()
        to_encode["reference_id"] = redis_data["reference_id"]
        to_encode["jti"] = jti
        expire = datetime.now(UTC) + timedelta(minutes=10)
        to_encode["expire"] = str(expire)
        await ctx.redis.expire(jti,600)
        jwt_token = jwt.encode(to_encode,key=SECRET_KEY,algorithm=ALGORITHM)
    except Exception as e:
        logging.error(f"Error in generate token {e}")
        raise HTTPException(500)
    return TokenResponse(token = jwt_token)

@app.get("/data")
async def get_data_by_jti(jwt_payload=Depends(verify_jwt)):
    """
    Retrieve stored OCR and match result from Redis using the jti as key.
    """
    jti = jwt_payload["jti"]
    if not await ctx.redis.exists(jti):
        raise HTTPException(status_code=404, detail="Data not found")

    data = await ctx.redis.hgetall(jti)

    # Optional: parse ocr_data back into dict
    if "ocr_data" in data:
        try:
            data["ocr_data"] = json.loads(data["ocr_data"])
        except json.JSONDecodeError:
            pass  # Keep as string if not parseable

    return data


@app.websocket("/api/v1/liveness")
async def websocket_liveness(websocket: WebSocket):
    """WebSocket endpoint for liveness detection"""
    
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return
    try:
        jwt_payload = verify_jwt_socket(token)
    except Exception as e:
        logging.error(f"Error in verify_jwt_socket {e}")
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()
    logging.info("WebSocket connection established") 
    if not await ctx.redis.exists(jwt_payload['jti']):
        raise WebSocketDisconnect(code = 3000, reason="Token invalid or expired")

    if await ctx.redis.hget(jwt_payload["jti"], "liveness_status") == 1:
        await websocket.send_json(
            {"status": "error", "message": "Liveness check already completed"}
        )
        await websocket.close()
        return

    steps = ["straight"]
    step = 0
    max_steps = len(steps)
    saved_frame = None

    try:
        while step < max_steps:
            current_step = steps[step]
            jti = jwt_payload["jti"]
            await websocket.send_json(
                {"status": "waiting", "gesture": current_step, "step": step}
            )  # TODO
            frame_data = await websocket.receive_bytes()

            if not minio_utils.validate_image(frame_data):
                await websocket.send_json(
                    {"status": "error", "message": "Invalid image format"}
                )
                continue

            detected_gesture = await send_image_via_grpc(frame_data, current_step)
            logging.info(
                f"Detected gesture: {detected_gesture}, Expected: {current_step}"
            )

            if detected_gesture:
                if step == 0:
                    saved_frame = frame_data
                    await minio_utils.upload_image_to_minio_async(saved_frame, f"liveness/{jti}/selfie_img.jpg")
                    
                await websocket.send_json({"status": "success", "gesture": current_step, "step": step})
                step += 1
            else:
                await websocket.send_json(
                    {"status": "error", "message": "Gesture mismatch"}
                )

        

        await ctx.redis.hset(jwt_payload["jti"], "liveness_status", 1)  
        await websocket.send_json(
            {"status": "completed", "message": "All gestures completed successfully"}
        )
        await websocket.close()

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {type(e).__name__}: {str(e)}")
        await websocket.close(code=1011, reason="Internal server error")


@app.post("/api/v1/submit_id")
async def submit_id(
    id_card: UploadFile = File(...),
    jwt_payload=Depends(verify_jwt),
    db=Depends(get_db),
):
    """Endpoint to submit ID card after liveness check"""
    
    if not await ctx.redis.exists(jwt_payload['jti']):
        raise HTTPException(
            status_code=401, detail="Invalid or expired token"
        )
    
    live = await ctx.redis.hgetall(jwt_payload["jti"])
    logging.info(f"Live status: {live}")
    if live["liveness_status"] != "1":
        raise HTTPException(
            status_code=400,
            detail="Liveness check not completed. Please complete the liveness check first.",
        )

    if live["matching_status"] == "1":
        raise HTTPException(
            status_code=400,
            detail="ID card already submitted. Please wait for validation.",
        )
    await minio_utils.upload_image_to_minio_async(
        id_card.file.read(), f"liveness/{jwt_payload['jti']}/id_card.jpg"
    )

    data = {
        "id_card_path": f"liveness/{jwt_payload['jti']}/id_card.jpg",
        "selfie_path": f"liveness/{jwt_payload['jti']}/selfie_img.jpg",
        "reference_id": jwt_payload["reference_id"],
        "jti": jwt_payload["jti"],
    }
    await ctx.rabbitmq.publish_request(data)

    message = {
        "reference_id": jwt_payload["reference_id"],
        "status": "pending",
        "error": None,
        "timestamp": int(time.time()),
    }
    await send_callback_raw(message, live['callback_url'])
    await ctx.redis.hset(jwt_payload["jti"], "status", "submitted")

    return {"message": "ID card submitted successfully"}




@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=4121, reload=False)
