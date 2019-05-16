import configparser
import os
import time
from concurrent import futures

import numpy as np
from mss import mss

from objects.Image import Image
from objects.Log import Log
from objects.Message import Message
from objects.MessageQueue import MessageQueue

from globals import MAX_WORKER, COMP_REQUEST, LANE_REQUEST, OBST_REQUEST, SIGN_REQUEST


class Capture(object):
    def __init__(self):
        self.log = Log('capture')
        self.mq = MessageQueue()
        self.queue_list = [COMP_REQUEST, LANE_REQUEST, OBST_REQUEST, SIGN_REQUEST]
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=MAX_WORKER)
        self.screen_shot = None

    def capture(self):
        self.screen_shot = Image(np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720})))

    def publish(self, queue, data):
        self.mq.publish(queue, Message(200, 'success', data).json())

    def publish_concurrent(self):
        data = self.screen_shot.message()
        try:
            [self.thread_pool.submit(self.publish, queue, data) for queue in self.queue_list]
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
