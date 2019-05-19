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
        self.screen_shot = \
            cv2.cvtColor(np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720})), cv2.COLOR_BGR2RGB)

    def publish(self, queue, data, callback_queue, corr_id=str(time.time())):
        self.mq.publish(queue, data, callback_queue, corr_id)

    def publish_concurrent(self):
        data = cv2.imencode('.jpg', self.screen_shot)[1].tostring()
        corr_id = str(time.time())
        try:
            self.publish(MQ.COMP_REQUEST, data, MQ.COMP_RESPONSE, corr_id)
            self.publish(MQ.LANE_REQUEST, data, MQ.LANE_RESPONSE, corr_id)
            self.publish(MQ.OBST_REQUEST, data, MQ.OBST_RESPONSE, corr_id)
            self.publish(MQ.SIGN_REQUEST, data, MQ.SIGN_RESPONSE, corr_id)
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
