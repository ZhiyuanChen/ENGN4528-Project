import base64
import time
import uuid
from concurrent import futures

import cv2
import numpy as np
from mss import mss

from globals import Message, MessageQueue, Log, MAX_WORKER, COMP_REQUEST, COMP_RESPONSE, \
    LANE_REQUEST, LANE_RESPONSE, OBST_REQUEST, OBST_RESPONSE, SIGN_REQUEST, SIGN_RESPONSE


class Capture(object):
    def __init__(self):
        self.log = Log('capture')
        self.mq = MessageQueue()
        self.queue_list = [COMP_REQUEST, LANE_REQUEST, OBST_REQUEST, SIGN_REQUEST]
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=MAX_WORKER)
        self.screen_shot = None

    def capture(self):
        self.screen_shot = np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720}))

    def publish(self, queue, data, callback_queue, corr_id=str(uuid.uuid4())):
        self.mq.publish(queue, Message(200, 'success', data).json(), callback_queue, corr_id)

    def publish_concurrent(self):
        data = base64.b64encode(cv2.imencode('.jpg', self.screen_shot)[1]).decode('ascii')
        try:
            self.publish(COMP_REQUEST, data, COMP_RESPONSE)
            self.publish(LANE_REQUEST, data, LANE_RESPONSE)
            self.publish(OBST_REQUEST, data, OBST_RESPONSE)
            self.publish(SIGN_REQUEST, data, SIGN_RESPONSE)
        except Exception as err:
            self.log.error(err)

    def process(self):
        t = time.time()
        self.capture()
        self.publish_concurrent()
        print(time.time()-t)


def main():
    capture = Capture()
    while(True):
        capture.process()


if __name__ == "__main__":
    main()
