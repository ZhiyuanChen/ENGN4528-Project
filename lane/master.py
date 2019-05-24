from lane import *

from globals import Master, Vernie, traceback, load_image, MQ, NN, cv2


class LaneMaster(Master):
    def __init__(self):
        super(LaneMaster, self).__init__(MQ.LANE_REQUEST)
        self.model = SCNN(pretrained=False)
        self.model.load_state_dict(torch.load(NN.LANE_WEIGHTS_PATH, map_location='cpu')['net'])
        self.model.eval()
        self.mq.channel.basic_consume(queue=MQ.LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        try:
            self.log.info(method.routing_key + ' received message')
            image = cv2.cvtColor(load_image(body), cv2.COLOR_BGR2RGB)
            x = transform(image)[0]
            x.unsqueeze_(0)
            result = self.model(x)[:2]
            image = draw_result(image, result)
            data = cv2.imencode('.jpg', image)[1].tostring()
            self.mq.publish(MQ.LANE_RESPONSE, data)
        except Exception as err:
            self.log.error(err)


def main():
    master = LaneMaster()


if __name__ == "__main__":
    main()
