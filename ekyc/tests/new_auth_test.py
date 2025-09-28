
import requests
import json
import uuid

# Initialize the data dictionary
data = {
    "reference_id": str(uuid.uuid4())
}
bearer_token = "DxOUUfD8j2ASUHXwV9z9E0wzcvYbOpPkFWhU6Xlr0ruaDGKdtjWVk6SboWFY5W5k"

# Define the Bearer token

# Set headers with Bearer token
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json"  # Explicitly set content type
}

try:
    # Make the POST request with headers
    response = requests.post(
        "http://localhost:5000/api/v1/create_session",
        json=data,
        headers=headers
    )
    print(response.json())
    # Check if the request was successful
    response.raise_for_status()
    new_data = {
        "session_id" : response.json()['session_id']
    }
    response = requests.post(
        "http://localhost:5000/api/v1/token",
        json=new_data,
        headers=headers
    )
    # Print the response data
    print(response.json())  # Assuming the response is JSON

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Is it running at http://localhost:5000?")
except requests.exceptions.RequestException as err:
    print(f"An error occurred: {err}")
except ValueError:
    print("Error: Response is not valid JSON")