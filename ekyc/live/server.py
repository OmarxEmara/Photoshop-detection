import grpc
import sys
import os
import cv2
import numpy as np
from concurrent import futures
import proto.liveness_pb2 as liveness_pb2
import proto.liveness_pb2_grpc as liveness_pb2_grpc
from liveness_utils import process_frame

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
instructions = ["left", "right", "straight", "smile", "blink"]


class LivenessService(liveness_pb2_grpc.LivenessServiceServicer):
    def DetectImage(self, request, context):
        image_data = request.image_data
        instruction = request.instruction
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return liveness_pb2.LivenessResponse(
                match=False, success=False, message="Invalid image data"
            )
        result = process_frame(frame, instruction)
        return liveness_pb2.LivenessResponse(
            match=result, success=True, message="Gesture classified successfully"
        )


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    liveness_pb2_grpc.add_LivenessServiceServicer_to_server(LivenessService(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    print("gRPC server started on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    start_grpc_server()
