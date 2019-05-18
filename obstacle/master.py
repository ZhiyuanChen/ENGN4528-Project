from globals import Master, Vernie, traceback, load_image, OBST_REQUEST, cv2


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(OBST_REQUEST)
        self.mq.channel.basic_consume(queue=OBST_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        image = load_image(body)
        cv2.imshow('images', image)
        cv2.waitKey(0)


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
