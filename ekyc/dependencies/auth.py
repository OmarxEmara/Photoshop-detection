from fastapi import Depends, HTTPException, status, Query, WebSocket, WebSocketException
from jose import jwt
from models.validation import Token
from utils.config import Config
from utils.logging_utils import get_custom_logger
from db.database import get_db
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated


logger = get_custom_logger("./auth.logs")
config = Config()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_token(token: str = Depends(oauth2_scheme)):
    """
    Security dependency that verifies the provided token against a known secure token.
    Raises HTTP 401 UNAUTHORIZED if verification fails.
    """
    correct_token = config.API_TOKEN
    if token != correct_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )


def verify_jwt_socket(token: str):
    try:
        payload = jwt.decode(token, key=config.SECRET_KEY, algorithms=config.AlGORITHM)
        logger.info(f"jwt {payload}")
    except Exception as e:
        logger.error(e)
        raise e
    return payload


def verify_jwt(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, key=config.SECRET_KEY, algorithms=config.AlGORITHM)
        logger.info(f"jwt {payload}")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    # db = next(get_db())
    # db_token = (
    #     db.query(Token).filter(Token.reference_id == payload["reference_id"]).first()
    # )
    # if not db_token:
    #     raise HTTPException(
    #         status_code=401, detail="Reference id not found in database"
    #     )

    # if db_token.usage_count >= 3:
    #     raise HTTPException(status_code=403, detail="Token usage limit exceeded")

    # try:
    #     db_token.usage_count += 1
    #     db.commit()
    # except Exception as e:
    #     db.rollback()
    #     logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
    #     raise HTTPException(status_code=500, detail="Database error occurred")

    return payload
