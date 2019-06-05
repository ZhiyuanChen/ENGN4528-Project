from globals import Master, load_image, MQ, cv2


class TheMaster(Master):
    def __init__(self):
        super(TheMaster, self).__init__()
        self.mq.channel.basic_consume(queue=MQ.COMP_RESPONSE, on_message_callback=self.comp_process)
        self.mq.channel.basic_consume(queue=MQ.LANE_RESPONSE, on_message_callback=self.lane_process)
        self.mq.channel.basic_consume(queue=MQ.OBST_RESPONSE, on_message_callback=self.obst_process)
        # self.mq.channel.basic_consume(queue=MQ.SIGN_RESPONSE, on_message_callback=self.sign_process)
        self.comp_window = self.window('Comprehensive', (850, 200))
        self.lane_window = self.window('Lane Line', (426, 340))
        self.obst_window = self.window('Obstacle', (426, 340))
        # self.comp_window = self.window('Lane')
        self.mq.channel.start_consuming()

    def comp_process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.comp_window, image)
            cv2.waitKey(1)
        except Exception as err:
            self.log.error(err)

    def lane_process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.lane_window, image)
            cv2.waitKey(1)
        except Exception as err:
            self.log.error(err)

    def obst_process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.obst_window, image)
            cv2.waitKey(1)
        except Exception as err:
            self.log.error(err)

    def sign_process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.sign_window, image)
            cv2.waitKey(1)
        except Exception as err:
            self.log.error(err)

    @staticmethod
    def window(window_name='image', window_size=(640, 360)):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        return window_name


def main():
    master = TheMaster()


if __name__ == "__main__":
    main()
