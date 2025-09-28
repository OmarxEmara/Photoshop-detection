import asyncio
import websockets
import json
from dotenv import load_dotenv
import os

# Path to the JPEG image to send
IMAGE_PATH = "/home/youssef-kabodan/Code/eKYC/ekyc/tests/Image-1.png"
load_dotenv()


async def send_frame(ws):
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        await ws.send(image_bytes)


async def test_liveness():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyZWZlcmVuY2VfaWQiOiJhMTY5Y2Y2OC02OTRlLTRiM2ItYjdjYS1mNDA5NDE5ODhmYjciLCJqdGkiOiJiZTJjZjljMi05YWEwLTQ0OWItOTAwMS0xMjM3NTZlYzFjMzYiLCJleHBpcmUiOiIyMDI1LTA2LTMwIDIxOjU5OjA0LjAzOTkwNCswMDowMCJ9.sRCmTv6Rs84Wj0gdy1fZW8Vs0DAk4W2bOSvM3C874Yg"
    uri = f"ws://localhost:5000/api/v1/liveness?token={token}"
    headers = {"Authorization": f"Bearer {token}"}
    async with websockets.connect(uri, additional_headers=headers) as ws:
        for i in range(3):  # Send 3 frames
            message = await ws.recv()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("Received non-JSON response:", message)
                continue

            print("Server:", data)

            status = data.get("status")

            if status == "waiting":
                print(f"Sending frame for gesture: {data['gesture']}")
                await send_frame(ws)
            elif status == "success":
                continue
            elif status == "error":
                print("Gesture mismatch, retrying...")
                await send_frame(ws)
            elif status == "completed":
                print("Liveness check completed.")
                break


if __name__ == "__main__":
    asyncio.run(test_liveness())
