import time
from concurrent import futures

import cv2
import numpy as np
from mss import mss

from globals import MessageQueue, Log, MQ, CONCURRENT


class Capture(object):
    def __init__(self):
        self.log = Log('capture')
        self.mq = MessageQueue()
        self.queue_list = [MQ.COMP_REQUEST, MQ.LANE_REQUEST, MQ.OBST_REQUEST, MQ.SIGN_REQUEST]
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=CONCURRENT.MAX_WORKER)
        self.screen_shot = None

    def capture(self):
        self.screen_shot = np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720}))

    def publish(self, queue, message):
        self.mq.publish(queue, message)

    def publish_concurrent(self):
        windshield = cv2.imencode('.jpg', self.screen_shot[0:490, 0:1200])[1].tostring()
        dashboard = cv2.imencode('.jpg', self.screen_shot[570:700, 622:1182])[1].tostring()
        try:
            self.publish(MQ.COMP_REQUEST, dashboard)
            self.publish(MQ.LANE_REQUEST, windshield)
            self.publish(MQ.OBST_REQUEST, windshield)
            print(time.time())
            # self.publish(MQ.SIGN_REQUEST, data)
        except Exception as err:
            self.log.error(err)

    def process(self):
        try:
            self.capture()
            self.publish_concurrent()
        except Exception as err:
            self.log.error(err)


def main():
    capture = Capture()
    for item in range(100):
        capture.process()


if __name__ == "__main__":
    main()
