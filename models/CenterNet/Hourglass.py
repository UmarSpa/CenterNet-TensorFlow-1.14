import tensorflow as tf
from .utils import conv_block, residual_block


class Pre(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(Pre, self).__init__()
        self.conv_block = conv_block(out_dim=128, kernel=7, stride=2, padding='same',
                                     weights_init=None if weights_dic is None else weights_dic['conv_block'])
        self.residual_block = residual_block(out_dim=256, stride=2, skip_flag=True,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'residual_block'])

    def call(self, inputs, training=None, mask=None):
        x = self.conv_block(inputs, training=training)
        x = self.residual_block(x, training=training)
        return x


class HourGlass52(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(HourGlass52, self).__init__()
        self.HG_module = HG_module(weights_dic)

    def call(self, inputs, training=None, mask=None):
        return self.HG_module(inputs, training=training)


class HG_module(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(HG_module, self).__init__()
        self.up1_0_residual = residual_block(out_dim=256, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up1_0_residual'])
        self.up1_1_residual = residual_block(out_dim=256, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up1_1_residual'])

        self.low1_0_residual = residual_block(out_dim=256, stride=2, skip_flag=True,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low1_0_residual'])
        self.low1_1_residual = residual_block(out_dim=256, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low1_1_residual'])

        self.up2_0_residual = residual_block(out_dim=256, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up2_0_residual'])
        self.up2_1_residual = residual_block(out_dim=256, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up2_1_residual'])

        self.low2_0_residual = residual_block(out_dim=384, stride=2, skip_flag=True,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low2_0_residual'])
        self.low2_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low2_1_residual'])

        self.up3_0_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up3_0_residual'])
        self.up3_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up3_1_residual'])

        self.low3_0_residual = residual_block(out_dim=384, stride=2, skip_flag=True,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low3_0_residual'])
        self.low3_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low3_1_residual'])

        self.up4_0_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up4_0_residual'])
        self.up4_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up4_1_residual'])

        self.low4_0_residual = residual_block(out_dim=384, stride=2, skip_flag=True,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low4_0_residual'])
        self.low4_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low4_1_residual'])

        self.up5_0_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up5_0_residual'])
        self.up5_1_residual = residual_block(out_dim=384, stride=1, skip_flag=False,
                                             weights_init=None if weights_dic is None else weights_dic[
                                                 'up5_1_residual'])

        self.low5_0_residual = residual_block(out_dim=512, stride=2, skip_flag=True,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low5_0_residual'])
        self.low5_1_residual = residual_block(out_dim=512, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low5_1_residual'])

        self.low6_0_residual = residual_block(out_dim=512, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low6_0_residual'])
        self.low6_1_residual = residual_block(out_dim=512, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low6_1_residual'])
        self.low6_2_residual = residual_block(out_dim=512, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low6_2_residual'])
        self.low6_3_residual = residual_block(out_dim=512, stride=1, skip_flag=False,
                                              weights_init=None if weights_dic is None else weights_dic[
                                                  'low6_3_residual'])

        self.low5_0_residual_b = residual_block(out_dim=512, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low5_0_residual_b'])
        self.low5_1_residual_b = residual_block(out_dim=384, stride=1, skip_flag=True,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low5_1_residual_b'])

        self.up5_residual_b = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')

        self.low4_0_residual_b = residual_block(out_dim=384, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low4_0_residual_b'])
        self.low4_1_residual_b = residual_block(out_dim=384, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low4_1_residual_b'])

        self.up4_residual_b = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')

        self.low3_0_residual_b = residual_block(out_dim=384, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low3_0_residual_b'])
        self.low3_1_residual_b = residual_block(out_dim=384, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low3_1_residual_b'])

        self.up3_residual_b = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')

        self.low2_0_residual_b = residual_block(out_dim=384, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low2_0_residual_b'])
        self.low2_1_residual_b = residual_block(out_dim=256, stride=1, skip_flag=True,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low2_1_residual_b'])

        self.up2_residual_b = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')

        self.low1_0_residual_b = residual_block(out_dim=256, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low1_0_residual_b'])
        self.low1_1_residual_b = residual_block(out_dim=256, stride=1, skip_flag=False,
                                                weights_init=None if weights_dic is None else weights_dic[
                                                    'low1_1_residual_b'])

        self.up1_residual_b = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')

    def call(self, inputs, training=None, mask=None):
        up1 = self.up1_0_residual(inputs, training=training)
        up1 = self.up1_1_residual(up1, training=training)

        low1 = self.low1_0_residual(inputs, training=training)
        low1 = self.low1_1_residual(low1, training=training)

        up2 = self.up2_0_residual(low1, training=training)
        up2 = self.up2_1_residual(up2, training=training)

        low2 = self.low2_0_residual(low1, training=training)
        low2 = self.low2_1_residual(low2, training=training)

        up3 = self.up3_0_residual(low2, training=training)
        up3 = self.up3_1_residual(up3, training=training)

        low3 = self.low3_0_residual(low2, training=training)
        low3 = self.low3_1_residual(low3, training=training)

        up4 = self.up4_0_residual(low3, training=training)
        up4 = self.up4_1_residual(up4, training=training)

        low4 = self.low4_0_residual(low3, training=training)
        low4 = self.low4_1_residual(low4, training=training)

        up5 = self.up5_0_residual(low4, training=training)
        up5 = self.up5_1_residual(up5, training=training)

        low5 = self.low5_0_residual(low4, training=training)
        low5 = self.low5_1_residual(low5, training=training)

        low6 = self.low6_0_residual(low5, training=training)
        low6 = self.low6_1_residual(low6, training=training)
        low6 = self.low6_2_residual(low6, training=training)
        low6 = self.low6_3_residual(low6, training=training)

        low5_b = self.low5_0_residual_b(low6, training=training)
        low5_b = self.low5_1_residual_b(low5_b, training=training)

        up5_b = self.up5_residual_b(low5_b)
        up5 += up5_b

        low4_b = self.low4_0_residual_b(up5, training=training)
        low4_b = self.low4_1_residual_b(low4_b, training=training)

        up4_b = self.up4_residual_b(low4_b)
        up4 += up4_b

        low3_b = self.low3_0_residual_b(up4, training=training)
        low3_b = self.low3_1_residual_b(low3_b, training=training)

        up3_b = self.up3_residual_b(low3_b)
        up3 += up3_b

        low2_b = self.low2_0_residual_b(up3, training=training)
        low2_b = self.low2_1_residual_b(low2_b, training=training)

        up2_b = self.up2_residual_b(low2_b)
        up2 += up2_b

        low1_b = self.low1_0_residual_b(up2, training=training)
        low1_b = self.low1_1_residual_b(low1_b, training=training)

        up1_b = self.up1_residual_b(low1_b)
        up1 += up1_b

        return up1
