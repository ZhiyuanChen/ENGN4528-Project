import cv2
import numpy as np
import json


def load_message(message):
    try:
        message = json.loads(message)
    except Exception:
        pass
    return message['code'], message['message'], message['data']
