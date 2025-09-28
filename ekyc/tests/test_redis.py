import redis

# Connect to Redis (adjust host/port/db if needed)
r = redis.Redis(host='localhost', port=6379, db=0)

# Replace 'myhash' with your hash name

hash_name = '7a9bf168-3fff-4bac-bdc2-7433e2c7ed67'

# Get all fields and values from the hash
hash_data = r.hgetall(hash_name)

# Convert bytes to strings (optional but common)
decoded_data = {k.decode(): v.decode() for k, v in hash_data.items()}

print(decoded_data)
