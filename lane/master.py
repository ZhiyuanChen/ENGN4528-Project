from lane import *

from globals import Master, Vernie, traceback, load_image, MQ, NN, cv2, time


class LaneMaster(Master):
    def __init__(self):
        super(LaneMaster, self).__init__(MQ.LANE_REQUEST)
        # self.model = SCNN(pretrained=False)
        # self.model.load_state_dict(torch.load(NN.LANE_WEIGHTS_PATH, map_location='cpu')['net'])
        # self.model.eval()
        self.mq.channel.basic_consume(queue=MQ.LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            #self.log.info(method.routing_key + ' received message')
            image = cv2.cvtColor(load_image(body), cv2.COLOR_BGR2RGB)
            print(time.time())
            # x = transform(image)[0]
            # x.unsqueeze_(0)
            # result = self.model(x)[:2]
            # image = self.draw_result(image, result)
            data = cv2.imencode('.jpg', image)[1].tostring()
            self.mq.publish(MQ.LANE_RESPONSE, data)
        except Exception as err:
            self.log.error(err)

    @staticmethod
    def draw_result(image, result):
        seg_pred = result[0].detach().cpu().numpy()
        exist_pred = result[1].detach().cpu().numpy()
        seg_pred = seg_pred[0]
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (800, 288))
        lane_img = np.zeros_like(image)
        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        coord_mask = np.argmax(seg_pred, axis=0)
        for i in range(0, 4):
            if exist_pred[0, i] > 0.5:
                lane_img[coord_mask == (i + 1)] = color[i]
        image = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=image, beta=1., gamma=0.)
        return image


def main():
    master = LaneMaster()


if __name__ == "__main__":
    main()
