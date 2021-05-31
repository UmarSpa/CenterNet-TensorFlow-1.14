import tensorflow as tf
from .utils import conv_block, convolution, batch_norm


def look_right_fn(input):
    input = tf.transpose(input, [0, 2, 3, 1])
    output = tf.pad(input, tf.constant([[0, 0], [0, 0], [0, input.get_shape().as_list()[2] - 1], [0, 0]]),
                    constant_values=tf.reduce_min(input))
    output = tf.nn.max_pool(output, ksize=(1, input.get_shape().as_list()[2]), strides=(1, 1), padding='VALID')
    return tf.transpose(output, [0, 3, 1, 2])


def look_down_fn(input):
    input = tf.transpose(input, [0, 2, 3, 1])
    output = tf.pad(input, tf.constant([[0, 0], [0, input.get_shape().as_list()[1] - 1], [0, 0], [0, 0]]),
                    constant_values=tf.reduce_min(input))
    output = tf.nn.max_pool(output, ksize=(input.get_shape().as_list()[1], 1), strides=(1, 1), padding='VALID')
    return tf.transpose(output, [0, 3, 1, 2])


def look_up_fn(input):
    input = tf.transpose(input, [0, 2, 3, 1])
    output = tf.pad(input, tf.constant([[0, 0], [input.get_shape().as_list()[1] - 1, 0], [0, 0], [0, 0]]),
                    constant_values=tf.reduce_min(input))
    output = tf.nn.max_pool(output, ksize=(input.get_shape().as_list()[1], 1), strides=(1, 1), padding='VALID')
    return tf.transpose(output, [0, 3, 1, 2])


def look_left_fn(input):
    input = tf.transpose(input, [0, 2, 3, 1])
    output = tf.pad(input, tf.constant([[0, 0], [0, 0], [input.get_shape().as_list()[2] - 1, 0], [0, 0]]),
                    constant_values=tf.reduce_min(input))
    output = tf.nn.max_pool(output, ksize=(1, input.get_shape().as_list()[2]), strides=(1, 1), padding='VALID')
    return tf.transpose(output, [0, 3, 1, 2])


class tl_pool(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(tl_pool, self).__init__()
        self.tl_pool = Pool(look_down_fn, look_right_fn, weights_dic=None if weights_dic is None else weights_dic)

    def call(self, inputs, training=None, mask=None):
        return self.tl_pool(inputs, training=training)


class br_pool(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(br_pool, self).__init__()
        self.br_pool = Pool(look_up_fn, look_left_fn, weights_dic=None if weights_dic is None else weights_dic)

    def call(self, inputs, training=None, mask=None):
        return self.br_pool(inputs, training=training)


class ct_pool(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(ct_pool, self).__init__()
        self.ct_pool = Pool_cross(look_down_fn, look_right_fn, look_up_fn, look_left_fn,
                                  weights_dic=None if weights_dic is None else weights_dic)

    def call(self, inputs, training=None, mask=None):
        return self.ct_pool(inputs, training=training)


class Pool(tf.keras.Model):
    def __init__(self, pool1, pool2, weights_dic=None):
        super(Pool, self).__init__()
        self.p1_conv1 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p1_conv1'])
        self.p2_conv1 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p2_conv1'])

        self.p_conv1 = convolution(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p_conv1'])
        self.p_bn1 = batch_norm(weights_init=None if weights_dic is None else weights_dic['p_bn1'])

        self.conv1 = convolution(out_dim=256, kernel=1, stride=1, padding='same',
                                 weights_init=None if weights_dic is None else weights_dic['conv1'])
        self.bn1 = batch_norm(weights_init=None if weights_dic is None else weights_dic['bn1'])

        self.conv2 = conv_block(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                weights_init=None if weights_dic is None else weights_dic['conv2'])

        self.pool1 = pool1
        self.pool2 = pool2

        self.look_conv1 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                     weights_init=None if weights_dic is None else weights_dic['look_conv1'])
        self.look_conv2 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                     weights_init=None if weights_dic is None else weights_dic['look_conv2'])

        self.P1_look_conv = convolution(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                        weights_init=None if weights_dic is None else weights_dic['P1_look_conv'])
        self.P2_look_conv = convolution(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                        weights_init=None if weights_dic is None else weights_dic['P2_look_conv'])

    def call(self, inputs, training=None, mask=None):
        look_conv1 = self.look_conv1(inputs, training=training)
        p1_conv1 = self.p1_conv1(inputs, training=training)
        look_right = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1 + look_right)
        pool1 = self.pool1(P1_look_conv)

        look_conv2 = self.look_conv2(inputs, training=training)
        p2_conv1 = self.p2_conv1(inputs, training=training)
        look_down = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1 + look_down)
        pool2 = self.pool2(P2_look_conv)

        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1, training=training)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(p_bn1 + bn1)

        conv2 = self.conv2(relu1, training=training)
        return conv2


class Pool_cross(tf.keras.Model):
    def __init__(self, pool1, pool2, pool3, pool4, weights_dic=None):
        super(Pool_cross, self).__init__()
        self.p1_conv1 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p1_conv1'])
        self.p2_conv1 = conv_block(out_dim=128, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p2_conv1'])

        self.p_conv1 = convolution(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   weights_init=None if weights_dic is None else weights_dic['p_conv1'])
        self.p_bn1 = batch_norm(weights_init=None if weights_dic is None else weights_dic['p_bn1'])

        self.conv1 = convolution(out_dim=256, kernel=1, stride=1, padding='same',
                                 weights_init=None if weights_dic is None else weights_dic['conv1'])
        self.bn1 = batch_norm(weights_init=None if weights_dic is None else weights_dic['bn1'])

        self.conv2 = conv_block(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                weights_init=None if weights_dic is None else weights_dic['conv2'])

        self.pool1 = pool1
        self.pool2 = pool2
        self.pool3 = pool3
        self.pool4 = pool4

    def call(self, inputs, training=None, mask=None):
        p1_conv1 = self.p1_conv1(inputs, training=training)
        pool1 = self.pool1(p1_conv1)
        pool1 = self.pool3(pool1)

        p2_conv1 = self.p2_conv1(inputs, training=training)
        pool2 = self.pool2(p2_conv1)
        pool2 = self.pool4(pool2)

        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1, training=training)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(p_bn1 + bn1)

        conv2 = self.conv2(relu1, training=training)
        return conv2
