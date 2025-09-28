import time
import requests
import aiohttp
from utils.config import Config
from utils.logging_utils import get_custom_logger
from schemas.validation_models import CallbackData
import hmac
import json
import hashlib

logger = get_custom_logger("./callback.logs")
config = Config()


async def send_callback(
    reference_id: str, ocr_data: dict, success: bool, match: bool, error, jti: str, callback_url: str
):
    timestamp = int(time.time())
    data = CallbackData(
        reference_id=str(reference_id),
        ocr_data=ocr_data,
        success=success,
        match=match,
        error=error,
        timestamp=timestamp,
    )

    body_json = json.dumps(data.model_dump(), separators=(",", ":"), ensure_ascii=False)
    body_bytes = body_json.encode("utf-8")

    signature = hmac.new(
        config.SECRET_KEY.encode("utf-8"), msg=body_bytes, digestmod=hashlib.sha256
    ).hexdigest()

    headers = {"Content-Type": "application/json", "X-Signature": signature}
    try:
        logger.info(callback_url)
    except Exception as e:
        logger.error(f"Failed to get callback URL from Redis: {e}")
        raise
    async with aiohttp.ClientSession() as session:
        async with session.post(
            callback_url, data=body_bytes, headers=headers
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to send callback: {response.status}")
            else:
                logger.info(
                    f"Sent data {data} to {callback_url} with HMAC and received {response.status}"
                )

    logger.info(f"Send to {callback_url} and received {response}")


async def send_callback_raw(data: dict, callback_url: str):
    """
    Send raw data to the callback URL.
    """
    try:

        body_json = json.dumps(data,separators=(",", ":"),ensure_ascii=False)
        body_bytes = body_json.encode("utf-8")
        signature = hmac.new(
        config.SECRET_KEY.encode("utf-8"), msg=body_bytes, digestmod=hashlib.sha256
    ).hexdigest()
        
        headers = {"Content-Type": "application/json", "X-Signature": signature}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url, data=body_bytes, headers=headers
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to send callback: {response.status}")
                else:
                    logger.info(
                        f"Sent data {data} to {callback_url} with HMAC and received {response.status}"
                    )
        return response
    except requests.RequestException as e:
        logger.error(f"Failed to send callback: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error while sending callback: {type(e).__name__}: {str(e)}"
        )
        raise
