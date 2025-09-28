from datetime import datetime, timedelta, UTC
import redis
import json


def check_redis_connection():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        expire = datetime.now(UTC) + timedelta(minutes=1)
        ttl = int((expire - datetime.now(UTC)).total_seconds())
        r.hset(
            "1234",
            mapping={
                "reference_id": "1234",
                "liveness_status": 0,
                "matching_status": 0,
                "expire": expire.timestamp(),
            },
        )
        r.expire("1234", 60)
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")


if __name__ == "__main__":
    # check_redis_connection()
    r = redis.Redis(host="localhost", port=6379, db=0)
    print(r.info())
    # print(r.hgetall("1234"))
    # print(r.ttl("1234"))
