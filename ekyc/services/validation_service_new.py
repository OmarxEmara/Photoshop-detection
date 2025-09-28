from message_service.message_service import RMQService
from services.matching_service import IDValidationPipeline
from services.ocr_service import detect_and_process_id_card

validation_pipeline = IDValidationPipeline()

rabbit_mq = RMQService()


def custom_callback():
    pass
