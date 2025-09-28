from message_service import RMQService
import json

rmq = RMQService()
rmq.connect()


for i  in range(1):
    print(f"Publishing message {i}")
    test_message = {
        "reference_id":"123",
        "jti": "test-jti-123",
        "selfie_path": "user-uploads/selfie3.png",
        "id_card_path": "user-uploads/id3.png"
    }

    rmq.publish("q-new-request", json.dumps(test_message), {
            'x-dead-letter-exchange': 'x-new-request-dead-letter'
        })
