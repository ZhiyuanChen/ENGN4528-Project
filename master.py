from globals import Master, load_image, MQ, cv2, time


class TheMaster(Master):
    def __init__(self):
        super(TheMaster, self).__init__()
        self.mq.channel.basic_consume(queue=MQ.COMP_RESPONSE, on_message_callback=self.comp_process)
        self.mq.channel.basic_consume(queue=MQ.LANE_RESPONSE, on_message_callback=self.lane_process)
        self.mq.channel.basic_consume(queue=MQ.OBST_RESPONSE, on_message_callback=self.obst_process)
        self.mq.channel.basic_consume(queue=MQ.SIGN_RESPONSE, on_message_callback=self.sign_process)
        self.comp_window = cv2.namedWindow('Comprehensive', 0)
        self.line_window = cv2.namedWindow('Lane Line', 0)
        self.obst_window = cv2.namedWindow('Obstacle', 0)
        self.sign_window = cv2.namedWindow('Traffic Sign', 0)
        self.mq.channel.start_consuming()

    def comp_process(self, ch, method, props, body):
        try:
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.comp_window, image)
            cv2.waitKey(1)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as err:
            self.log.error(err)

    def lane_process(self, ch, method, props, body):
        try:
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.line_window, image)
            cv2.waitKey(1)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as err:
            self.log.error(err)

    def obst_process(self, ch, method, props, body):
        try:
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.obst_window, image)
            cv2.waitKey(1)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as err:
            self.log.error(err)

    def sign_process(self, ch, method, props, body):
        try:
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            cv2.imshow(self.sign_window, image)
            cv2.waitKey(1)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as err:
            self.log.error(err)


def main():
    master = TheMaster()


if __name__ == "__main__":
    main()
