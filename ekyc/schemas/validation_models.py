from pydantic import BaseModel
from typing import Optional, Dict, Any


class CreateSessionRequest(BaseModel):
    reference_id: str
    callback_url: Optional[str] = None

class TokenRequest(BaseModel):
    session_id: str

class TokenResponse(BaseModel):
    token: str
    

class CreateSessionResponse(BaseModel):
    session_id: str
    expire_timestamp: int

class ValidationResponse(BaseModel):
    success: bool
    match: bool
    error: Optional[str] = None
    processing_time: Optional[float] = None
    FullName: Optional[str] = None
    NID: Optional[str] = None
    Address: Optional[str] = None
    Serial: Optional[str] = None


class CallbackData(BaseModel):
    reference_id: str
    ocr_data: Dict[str, Any]
    success: bool
    match: bool
    error: Optional[str]
    timestamp: int


class HealthResponse(BaseModel):
    status: str
    timestamp: int
