import numpy as np
import tensorflow as tf

layers = tf.keras.layers


def sigmoid(x):
    x = tf.sigmoid(x)
    x = tf.clip_by_value(x, clip_value_min=1e-4, clip_value_max=1 - 1e-4)
    return x


class kp_layer(tf.keras.Model):
    def __init__(self, outdim=80, weights_dic=None):
        super(kp_layer, self).__init__()
        self.conv1 = convolution(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]], bias=True,
                                 weights_init=None if weights_dic is None else weights_dic['conv1'])
        self.conv2 = convolution(out_dim=outdim, kernel=1, stride=1, padding='same', bias=True,
                                 weights_init=None if weights_dic is None else weights_dic['conv2'])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x


class convolution(tf.keras.Model):
    def __init__(self, out_dim, kernel, stride, padding, weights_init=None, bias=False):
        super(convolution, self).__init__()
        self.conv = layers.Conv2D(
            filters=out_dim,
            kernel_size=kernel,
            strides=(stride, stride),
            data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.001) if weights_init is None else tf.constant_initializer(
                np.transpose(weights_init['conv.weight'], [2, 3, 1, 0])),
            bias_initializer='zeros' if (bias is False or weights_init is None) else tf.constant_initializer(
                weights_init['conv.bias']),
            padding=padding,
            use_bias=bias
        )

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)


class batch_norm(tf.keras.Model):
    def __init__(self, weights_init=None):
        super(batch_norm, self).__init__()
        self.batch_norm = layers.BatchNormalization(
            axis=1,
            momentum=0.1,
            epsilon=1e-5,
            beta_initializer='zeros' if weights_init is None else tf.constant_initializer(weights_init['bn.bias']),
            gamma_initializer='ones' if weights_init is None else tf.constant_initializer(weights_init['bn.weight']),
            moving_mean_initializer='zeros' if weights_init is None else tf.constant_initializer(
                weights_init['bn.running_mean']),
            moving_variance_initializer='ones' if weights_init is None else tf.constant_initializer(
                weights_init['bn.running_var']),
        )

    def call(self, inputs, training=None, mask=None):
        return self.batch_norm(inputs, training=training)


class conv_block(tf.keras.Model):
    def __init__(self, out_dim, kernel, stride, padding, weights_init=None):
        super(conv_block, self).__init__()
        self.conv = layers.Conv2D(
            filters=out_dim,
            kernel_size=kernel,
            strides=(stride, stride),
            data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.001) if weights_init is None else tf.constant_initializer(
                np.transpose(weights_init['conv.weight'], [2, 3, 1, 0])),
            padding=padding,
            use_bias=False
        )
        self.batch_norm = layers.BatchNormalization(
            axis=1,
            momentum=0.1,
            epsilon=1e-5,
            beta_initializer='zeros' if weights_init is None else tf.constant_initializer(weights_init['bn.bias']),
            gamma_initializer='ones' if weights_init is None else tf.constant_initializer(weights_init['bn.weight']),
            moving_mean_initializer='zeros' if weights_init is None else tf.constant_initializer(
                weights_init['bn.running_mean']),
            moving_variance_initializer='ones' if weights_init is None else tf.constant_initializer(
                weights_init['bn.running_var']),
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = tf.nn.relu(x)
        return x


class residual_block(tf.keras.Model):
    def __init__(self, out_dim, stride, skip_flag, weights_init=None):
        super(residual_block, self).__init__()
        self.skip_flag = skip_flag
        self.conv1 = layers.Conv2D(
            filters=out_dim,
            kernel_size=3,
            strides=(stride, stride),
            data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.001) if weights_init is None else tf.constant_initializer(
                np.transpose(weights_init['conv1.weight'], [2, 3, 1, 0])),
            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            use_bias=False
        )
        self.batch_norm1 = layers.BatchNormalization(
            axis=1,
            momentum=0.1,
            epsilon=1e-5,
            beta_initializer='zeros' if weights_init is None else tf.constant_initializer(weights_init['bn1.bias']),
            gamma_initializer='ones' if weights_init is None else tf.constant_initializer(weights_init['bn1.weight']),
            moving_mean_initializer='zeros' if weights_init is None else tf.constant_initializer(
                weights_init['bn1.running_mean']),
            moving_variance_initializer='ones' if weights_init is None else tf.constant_initializer(
                weights_init['bn1.running_var']),
        )
        self.conv2 = layers.Conv2D(
            filters=out_dim,
            kernel_size=3,
            strides=(1, 1),
            data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.001) if weights_init is None else tf.constant_initializer(
                np.transpose(weights_init['conv2.weight'], [2, 3, 1, 0])),
            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            use_bias=False
        )
        self.batch_norm2 = layers.BatchNormalization(
            axis=1,
            momentum=0.1,
            epsilon=1e-5,
            beta_initializer='zeros' if weights_init is None else tf.constant_initializer(weights_init['bn2.bias']),
            gamma_initializer='ones' if weights_init is None else tf.constant_initializer(weights_init['bn2.weight']),
            moving_mean_initializer='zeros' if weights_init is None else tf.constant_initializer(
                weights_init['bn2.running_mean']),
            moving_variance_initializer='ones' if weights_init is None else tf.constant_initializer(
                weights_init['bn2.running_var']),
        )
        if self.skip_flag:
            self.skip_conv = layers.Conv2D(
                filters=out_dim,
                kernel_size=1,
                strides=(stride, stride),
                data_format="channels_first",
                kernel_initializer=tf.random_normal_initializer(
                    stddev=0.001) if weights_init is None else tf.constant_initializer(
                    np.transpose(weights_init['skip.0.weight'], [2, 3, 1, 0])),
                padding='valid',
                use_bias=False,
            )
            self.skip_bn = layers.BatchNormalization(
                axis=1,
                momentum=0.1,
                epsilon=1e-5,
                beta_initializer='zeros' if weights_init is None else tf.constant_initializer(
                    weights_init['skip.1.bias']),
                gamma_initializer='ones' if weights_init is None else tf.constant_initializer(
                    weights_init['skip.1.weight']),
                moving_mean_initializer='zeros' if weights_init is None else tf.constant_initializer(
                    weights_init['skip.1.running_mean']),
                moving_variance_initializer='ones' if weights_init is None else tf.constant_initializer(
                    weights_init['skip.1.running_var']),
            )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)

        if self.skip_flag:
            shortcut = self.skip_conv(inputs)
            shortcut = self.skip_bn(shortcut, training=training)
        else:
            shortcut = inputs

        x += shortcut
        return tf.nn.relu(x)


class my_transpose_2Dconv_block(tf.keras.Model):
    def __init__(self, filters, weight_path, weights_dict, conv_name, bn_name):
        super(my_transpose_2Dconv_block, self).__init__()

        self.tconv = tf.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation=None,
            use_bias=False,
            data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if weight_path is None else tf.constant_initializer(np.transpose(weights_dict[conv_name + '.weight'], (2, 3, 1, 0))),
            padding="same")
        # self.bn = tf.layers.BatchNormalization(momentum=0.1)
        self.bn = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=0.1,
            beta_initializer='zeros' if weight_path is None else tf.constant_initializer(weights_dict[bn_name + '.bias']),
            gamma_initializer='ones' if weight_path is None else tf.constant_initializer(weights_dict[bn_name + '.weight']),
            moving_mean_initializer='zeros' if weight_path is None else tf.constant_initializer(weights_dict[bn_name + '.running_mean']),
            moving_variance_initializer='ones' if weight_path is None else tf.constant_initializer(weights_dict[bn_name + '.running_var']))

    def call(self, x, is_training):
        x = self.tconv(x)
        x = self.bn(x, is_training)
        x = tf.nn.relu(x)
        return x
