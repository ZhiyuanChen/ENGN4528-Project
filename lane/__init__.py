# https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/SCNN-Tensorflow/lane-detection-model

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layer
from easydict import EasyDict as edict
from collections import OrderedDict
import math
__LANE = edict()
LANE = __LANE
# The shadownet training epochs
__LANE.EPOCHS = 90100  # 200010
# The display step
__LANE.DISPLAY_STEP = 1
# The test display step during training process
__LANE.TEST_DISPLAY_STEP = 1000
# The momentum parameter of the optimizer
__LANE.MOMENTUM = 0.9
# The initial learning rate
__LANE.LEARNING_RATE = 0.01  # 0.0005
# The GPU resource used during training process
__LANE.GPU_MEMORY_FRACTION = 0.85
# The GPU allow growth parameter during tensorflow training process
__LANE.TF_ALLOW_GROWTH = True
# The learning rate decay steps
__LANE.LR_DECAY_STEPS = 210000
# The learning rate decay rate
__LANE.LR_DECAY_RATE = 0.1
# The class numbers
__LANE.CLASSES_NUMS = 2
# The shape of the image
__LANE.IMG_HEIGHT = 288
__LANE.IMG_WIDTH = 800
# The batch size
__LANE.BATCH_SIZE = 1
__LANE.TRN_BATCH_SIZE = 8
__LANE.VAL_BATCH_SIZE = 8


class CNNBaseModel(object):
    """
    Base model for other specific cnn ctpn_models
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, is_training=None, noise_shape=None, name=None):
        """

        :param name:
        :param inputdata:
        :param keep_prob:
        :param noise_shape:
        :return:
        """
        def f1():
            """

            :return:
            """
            # print('batch_normalization: train phase')
            return tf.nn.dropout(inputdata, keep_prob, noise_shape, name=name)

        def f2():
            """

            :return:
            """
            # print('batch_normalization: test phase')
            return inputdata

        output = tf.cond(is_training, f1, f2)
        # output = tf.nn.dropout(inputdata, keep_prob, noise_shape, name=name)

        return output

        # return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: tf.identity(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training, name):
        """

        :param inputdata:
        :param is_training:
        :param name:
        :return:
        """
        def f1():
            """

            :return:
            """
            # print('batch_normalization: train phase')
            return tf_layer.batch_norm(
                             inputdata, is_training=True,
                             center=True, scale=True, updates_collections=None,
                             scope=name, reuse=False)

        def f2():
            """

            :return:
            """
            # print('batch_normalization: test phase')
            return tf_layer.batch_norm(
                             inputdata, is_training=False,
                             center=True, scale=True, updates_collections=None,
                             scope=name, reuse=True)

        output = tf.cond(is_training, f1, f2)

        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param activation: whether to apply a activation func to deconv result
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_tensor, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret


class VGG16Encoder(CNNBaseModel):
    """
    实现了一个基于vgg16的特征编码类
    """

    def __init__(self, phase):
        """

        :param phase:
        """
        super(VGG16Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _conv_dilated_stage(self, input_tensor, k_size, out_dims, name,
                            dilation=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.dilation_conv(input_tensor=input_tensor, out_dims=out_dims,
                                      k_size=k_size, rate=dilation,
                                      use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def encode(self, input_tensor, name):
        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :return: 输出vgg16编码特征
        """
        ret = OrderedDict()
        with tf.variable_scope(name):
            # conv stage 1_1
            conv_1_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                        out_dims=64, name='conv1_1')

            # conv stage 1_2
            conv_1_2 = self._conv_stage(input_tensor=conv_1_1, k_size=3,
                                        out_dims=64, name='conv1_2')

            # pool stage 1
            pool1 = self.maxpooling(inputdata=conv_1_2, kernel_size=2,
                                    stride=2, name='pool1')

            # conv stage 2_1
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3,
                                        out_dims=128, name='conv2_1')

            # conv stage 2_2
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3,
                                        out_dims=128, name='conv2_2')

            # pool stage 2
            pool2 = self.maxpooling(inputdata=conv_2_2, kernel_size=2,
                                    stride=2, name='pool2')

            # conv stage 3_1
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3,
                                        out_dims=256, name='conv3_1')

            # conv_stage 3_2
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3,
                                        out_dims=256, name='conv3_2')

            # conv stage 3_3
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3,
                                        out_dims=256, name='conv3_3')

            # pool stage 3
            pool3 = self.maxpooling(inputdata=conv_3_3, kernel_size=2,
                                    stride=2, name='pool3')

            # conv stage 4_1
            conv_4_1 = self._conv_stage(input_tensor=pool3, k_size=3,
                                        out_dims=512, name='conv4_1')

            # conv stage 4_2
            conv_4_2 = self._conv_stage(input_tensor=conv_4_1, k_size=3,
                                        out_dims=512, name='conv4_2')

            # conv stage 4_3
            conv_4_3 = self._conv_stage(input_tensor=conv_4_2, k_size=3,
                                        out_dims=512, name='conv4_3')

            ### add dilated convolution ###

            # conv stage 5_1
            conv_5_1 = self._conv_dilated_stage(input_tensor=conv_4_3, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_1')

            # conv stage 5_2
            conv_5_2 = self._conv_dilated_stage(input_tensor=conv_5_1, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_2')

            # conv stage 5_3
            conv_5_3 = self._conv_dilated_stage(input_tensor=conv_5_2, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_3')

            # added part of SCNN #

            # conv stage 5_4
            conv_5_4 = self._conv_dilated_stage(input_tensor=conv_5_3, k_size=3,
                                                out_dims=1024, dilation=4, name='conv5_4')

            # conv stage 5_5
            conv_5_5 = self._conv_stage(input_tensor=conv_5_4, k_size=1,
                                        out_dims=128, name='conv5_5')  # 8 x 36 x 100 x 128

            # add message passing #

            # top to down #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(conv_5_5.get_shape().as_list()[1]):
                feature_list_old.append(tf.expand_dims(conv_5_5[:, cnt, :, :], axis=1))
            feature_list_new.append(tf.expand_dims(conv_5_5[:, 0, :, :], axis=1))

            w1 = tf.get_variable('W1', [1, 9, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_1"):
                conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_1)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_1", reuse=True):
                    conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_1)

            # down to top #
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(LANE.IMG_HEIGHT / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w2 = tf.get_variable('W2', [1, 9, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_2"):
                conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_2)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_2", reuse=True):
                    conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_2)

            feature_list_new.reverse()

            processed_feature = tf.stack(feature_list_new, axis=1)
            processed_feature = tf.squeeze(processed_feature, axis=2)

            # left to right #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(processed_feature.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

            w3 = tf.get_variable('W3', [9, 1, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_3"):
                conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_3)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_3", reuse=True):
                    conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_3)

            # right to left #

            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(LANE.IMG_WIDTH / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w4 = tf.get_variable('W4', [9, 1, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_4"):
                conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_4)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_4", reuse=True):
                    conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_4)

            feature_list_new.reverse()
            processed_feature = tf.stack(feature_list_new, axis=2)
            processed_feature = tf.squeeze(processed_feature, axis=3)

            #######################

            dropout_output = self.dropout(processed_feature, 0.9, is_training=self._is_training,
                                          name='dropout')  # 0.9 denotes the probability of being kept

            conv_output = self.conv2d(inputdata=dropout_output, out_channel=5,
                                      kernel_size=1, use_bias=True, name='conv_6')

            ret['prob_output'] = tf.image.resize_images(conv_output, [LANE.IMG_HEIGHT, LANE.IMG_WIDTH])

            ### add lane existence prediction branch ###

            # spatial softmax #
            features = conv_output  # N x H x W x C
            softmax = tf.nn.softmax(features)

            avg_pool = self.avgpooling(softmax, kernel_size=2, stride=2)
            _, H, W, C = avg_pool.get_shape().as_list()
            reshape_output = tf.reshape(avg_pool, [-1, H * W * C])
            fc_output = self.fullyconnect(reshape_output, 128)
            relu_output = self.relu(inputdata=fc_output, name='relu6')
            fc_output = self.fullyconnect(relu_output, 4)
            existence_output = fc_output

            ret['existence_output'] = existence_output

        return ret


class LaneNet(CNNBaseModel):
    """
    Lane detection model
    """

    @staticmethod
    def inference(input_tensor, phase, name):
        """
        feed forward
        :param name:
        :param input_tensor:
        :param phase:
        :return:
        """
        with tf.variable_scope(name):
            with tf.variable_scope('inference'):
                encoder = VGG16Encoder(phase=phase)
                encode_ret = encoder.encode(input_tensor=input_tensor, name='encode')

            return encode_ret

    @staticmethod
    def test_inference(input_tensor, phase, name):
        inference_ret = LaneNet.inference(input_tensor, phase, name)
        with tf.variable_scope(name):
            # feed forward to obtain logits
            # Compute loss

            decode_logits = inference_ret['prob_output']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            prob_list = []
            kernel = tf.get_variable('kernel', [9, 9, 1, 1], initializer=tf.constant_initializer(1.0 / 81),
                                     trainable=False)

            with tf.variable_scope("convs_smooth"):
                prob_smooth = tf.nn.conv2d(tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, 0], axis=3), tf.float32),
                                           kernel, [1, 1, 1, 1], 'SAME')
                prob_list.append(prob_smooth)

            for cnt in range(1, binary_seg_ret.get_shape().as_list()[3]):
                with tf.variable_scope("convs_smooth", reuse=True):
                    prob_smooth = tf.nn.conv2d(
                        tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, cnt], axis=3), tf.float32), kernel, [1, 1, 1, 1],
                        'SAME')
                    prob_list.append(prob_smooth)
            processed_prob = tf.stack(prob_list, axis=4)
            processed_prob = tf.squeeze(processed_prob, axis=3)
            binary_seg_ret = processed_prob

            # Predict lane existence:
            existence_logit = inference_ret['existence_output']
            existence_output = tf.nn.sigmoid(existence_logit)

            return binary_seg_ret, existence_output

    @staticmethod
    def loss(inference, binary_label, existence_label, name):
        """
        :param name:
        :param inference:
        :param existence_label:
        :param binary_label:
        :return:
        """
        # feed forward to obtain logits

        with tf.variable_scope(name):

            inference_ret = inference

            # Compute the segmentation loss

            decode_logits = inference_ret['prob_output']
            decode_logits_reshape = tf.reshape(
                decode_logits,
                shape=[decode_logits.get_shape().as_list()[0],
                       decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
                       decode_logits.get_shape().as_list()[3]])

            binary_label_reshape = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0],
                       binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
            binary_label_reshape = tf.one_hot(binary_label_reshape, depth=5)
            class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
            weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
            binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                                       logits=decode_logits_reshape,
                                                                       weights=weights_loss)
            binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

            # Compute the sigmoid loss

            existence_logits = inference_ret['existence_output']
            existence_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=existence_label, logits=existence_logits)
            existence_loss = tf.reduce_mean(existence_loss)

        # Compute the overall loss

        total_loss = binary_segmentation_loss + 0.1 * existence_loss
        ret = {
            'total_loss': total_loss,
            'instance_seg_logits': decode_logits,
            'instance_seg_loss': binary_segmentation_loss,
            'existence_logits': existence_logits,
            'existence_pre_loss': existence_loss
        }

        tf.add_to_collection('total_loss', total_loss)
        tf.add_to_collection('instance_seg_logits', decode_logits)
        tf.add_to_collection('instance_seg_loss', binary_segmentation_loss)
        tf.add_to_collection('existence_logits', existence_logits)
        tf.add_to_collection('existence_pre_loss', existence_loss)

        return ret
