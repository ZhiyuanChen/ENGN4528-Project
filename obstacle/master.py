from globals import Master, load_image, MQ, NN, LOG, color_map
from obstacle import *


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(MQ.OBST_REQUEST)
        with tf.device("/gpu:0"):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=LOG.DIR, config=coco.CocoConfig())
        self.model.load_weights(NN.OBST_WEIGHTS_PATH, by_name=True)
        self.queue = MQ.OBST_REQUEST
        self.mq.channel.basic_consume(self.queue, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            result = self.model.detect([image], verbose=1)[0]
            image = self.draw_result(image, result)
            data = cv2.imencode('.jpg', image)[1].tostring()
            self.mq.publish(MQ.OBST_RESPONSE, data)
        except Exception as err:
            self.log.error(err)

    @staticmethod
    def draw_result(image, result, show_mask=True, show_bbox=True):
        for i in range(result['rois'].shape[0]):
            color = color_map()[result['class_ids'][i]].astype(np.int).tolist()
            if show_bbox:
                cord = result['rois'][i]
                cv2.rectangle(image, (cord[1], cord[0]), (cord[3], cord[2]), color, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, '{}: {:.3f}'.format(class_names[result['class_ids'][i] - 1], result['scores'][i]),
                            (cord[1], cord[0]), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            if show_mask:
                mask = result['masks'][:, :, i]
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int)
                color_mask[mask] = color
                image = cv2.addWeighted(color_mask, 0.5, image.astype(np.int), 1, 0)
        return image.astype('uint8')


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
