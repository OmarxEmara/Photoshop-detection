import concurrent.futures
import grpc
import proto.liveness_pb2 as liveness_pb2
import proto.liveness_pb2_grpc as liveness_pb2_grpc
import concurrent
import time


executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# For testing gRPC.


def send_image_via_grpc(jpeg_path: str, target_name: str):
    with open(jpeg_path, "rb") as f:
        jpeg_data = f.read()

    channel = grpc.insecure_channel("localhost:50052")
    stub = liveness_pb2_grpc.LivenessServiceStub(channel)

    response = stub.DetectImage(
        liveness_pb2.LivenessRequest(image_data=jpeg_data, instruction="right")
    )
    print("Server response:", response.match)


# Example:
# send_image_via_grpc("sample.jpg", "saved_sample.jpg")
if __name__ == "__main__":
    num_requests = 1000
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                send_image_via_grpc,
                "/home/youssef-kabodan/Code/eKYC/ekyc/tests/Image-2.png",
                i + 1,
            )
            for i in range(num_requests)
        ]
        durations = [future.result() for future in futures]
    total_time = time.time() - start_time
    print(f"Parallel Total Time: {total_time:.3f}s")
    print(f"Parallel Throughput: {num_requests / total_time:.2f} requests/sec")
