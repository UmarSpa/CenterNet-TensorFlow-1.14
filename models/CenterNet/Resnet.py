import tensorflow as tf
import pdb
import os
import numpy as np
from .utils import my_transpose_2Dconv_block
BN_MOMENTUM = 0.1

class Post_ResNet(tf.keras.Model):
    def __init__(self, backbone, weight_path):
        super(Post_ResNet, self).__init__()

        if backbone in ["R-50"]:
            t_filters = 256
        elif backbone in ["R-18"]:
            t_filters = 160
        else:
            raise NameError('myError: backbone name not as expected.')

        if weight_path is not None:
            assert os.path.isfile(weight_path), "Error, {} is not a file".format(weight_path)
            weights_dict = np.load(weight_path, allow_pickle=True)[()]

            # Delete non useful information
            for k in list(weights_dict.keys()):
                if "num_batches_tracked" in k:
                    del weights_dict[k]
            weights_dict = {".".join(k.split(".")[2:]): v for k, v in weights_dict.items()}

        else:
            weights_dict = None
            weight_path = None

        self.tconv1 = my_transpose_2Dconv_block(t_filters, weight_path, weights_dict, 'deconv_layers.0', 'deconv_layers.1')
        self.tconv2 = my_transpose_2Dconv_block(t_filters, weight_path, weights_dict, 'deconv_layers.3', 'deconv_layers.4')
        self.tconv3 = my_transpose_2Dconv_block(t_filters, weight_path, weights_dict, 'deconv_layers.6', 'deconv_layers.7')

    def call(self, x, training):
        x = self.tconv1(x, training)
        x = self.tconv2(x, training)
        x = self.tconv3(x, training)
        return x

class ResNet18(tf.keras.Model):

    def __init__(self, weights_fp=None):
        super(ResNet18, self).__init__()
        self.res18 = ResNet(
            "BasicBlock",
            [2, 2, 2, 2],
            weights_fp
        )

    def call(self, x, training=False):
        return self.res18(x, training)


class ResNet50(tf.keras.Model):

    def __init__(self, weights_fp=None):
        super(ResNet50, self).__init__()
        self.res50 = ResNet(
            "Bottleneck",
            [3, 4, 6, 3],
            weights_fp
        )

    def call(self, x, training=False):
        return self.res50(x, training)


class ResNet101(tf.keras.Model):

    def __init__(self, weights_fp=None):
        super(ResNet101, self).__init__()
        self.res101 = ResNet(
            "Bottleneck",
            [3, 4, 23, 3],
            weights_fp
        )

    def call(self, x, training=False):
        return self.res101(x, training)


class ResNet(tf.keras.Model):

    def __init__(self, block, layers, weights_fp=None):
        super(ResNet, self).__init__()
        block = eval(block)
        if weights_fp is not None:
            assert os.path.isfile(weights_fp), "Error, {} is not a file".format(weights_fp)
            weights_dict = np.load(weights_fp, allow_pickle=True)[()]

            # Delete non useful information
            for k in list(weights_dict.keys()):
                if "num_batches_tracked" in k:
                    del weights_dict[k]

            if 'posenet' in weights_fp or 'rootnet' in weights_fp:
                weights_dict = {".".join(k.split(".")[2:]): v for k, v in weights_dict.items()}
            else:  # ResNet50
                weights_dict = {".".join(k.split(".")): v for k, v in weights_dict.items()}
        else:
            weights_dict = None

        self.conv1 = tf.keras.layers.Conv2D(
            64, kernel_size=7, padding=[[0, 0], [0, 0], [3, 3], [3, 3]],
            strides=2, use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.001) if weights_fp is None else tf.constant_initializer(
                np.transpose(weights_dict['conv1.weight'], (2, 3, 1, 0)))
        )
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer='zeros' if weights_fp is None else tf.constant_initializer(weights_dict['bn1.bias']),
            gamma_initializer='ones' if weights_fp is None else tf.constant_initializer(weights_dict['bn1.weight']),
            moving_mean_initializer='zeros' if weights_fp is None else tf.constant_initializer(
                weights_dict['bn1.running_mean']),
            moving_variance_initializer='ones' if weights_fp is None else tf.constant_initializer(
                weights_dict['bn1.running_var']),
        )
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=(2, 2), padding='valid', data_format="channels_first")
        self.layer1 = ResNetLayer(64, block, 64, layers[0], init_weights=getSpecificWeights(weights_dict, "layer", 1))
        self.layer2 = ResNetLayer(64 * block.expansion, block, 128, layers[1], stride=2,
                                  init_weights=getSpecificWeights(weights_dict, "layer", 2))
        self.layer3 = ResNetLayer(128 * block.expansion, block, 256, layers[2], stride=2,
                                  init_weights=getSpecificWeights(weights_dict, "layer", 3))
        self.layer4 = ResNetLayer(256 * block.expansion, block, 512, layers[3], stride=2,
                                  init_weights=getSpecificWeights(weights_dict, "layer", 4))

    def call(self, x, is_training=False):
        x = self.conv1(x)
        x = self.bn1(x, is_training)
        x = self.relu(x)

        # We have to externally add a pad before the maxpool
        x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]])
        x = self.maxpool(x)  # we chnage the padding to valid now

        x = self.layer1(x, is_training)
        x = self.layer2(x, is_training)
        x = self.layer3(x, is_training)
        x = self.layer4(x, is_training)

        return x


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, init_weights=None):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            planes, kernel_size=1,
            strides=1, use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['conv1.weight'], (2, 3, 1, 0)))
        )
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn1.bias']),
            gamma_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn1.weight']),
            moving_mean_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn1.running_mean']),
            moving_variance_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn1.running_var'])
        )
        self.conv2 = tf.keras.layers.Conv2D(
            planes, kernel_size=3, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            strides=stride, use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['conv2.weight'], (2, 3, 1, 0)))
        )
        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn2.bias']),
            gamma_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn2.weight']),
            moving_mean_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn2.running_mean']),
            moving_variance_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn2.running_var']))
        self.conv3 = tf.keras.layers.Conv2D(
            planes * self.expansion, kernel_size=1,
            strides=1, use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['conv3.weight'], (2, 3, 1, 0)))
        )
        self.bn3 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn3.bias']),
            gamma_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn3.weight']),
            moving_mean_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['bn3.running_mean']),
            moving_variance_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['bn3.running_var']))
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, is_training=False):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, is_training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, is_training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, is_training)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, init_weights=None):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            planes, kernel_size=3,
            strides=stride, padding="same", use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['conv1.weight'], (2, 3, 1, 0)))
            )

        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer = 'zeros' if init_weights is None else tf.constant_initializer(init_weights['bn1.bias']),
            gamma_initializer = 'ones' if init_weights is None else tf.constant_initializer(init_weights['bn1.weight']),
            moving_mean_initializer = 'zeros' if init_weights is None else tf.constant_initializer(init_weights['bn1.running_mean']),
            moving_variance_initializer = 'ones' if init_weights is None else tf.constant_initializer(init_weights['bn1.running_var'])
            )

        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(
            planes, kernel_size=3,
            strides=1, padding="same", use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['conv2.weight'], (2, 3, 1, 0)))
            )

        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer = 'zeros' if init_weights is None else tf.constant_initializer(init_weights['bn2.bias']),
            gamma_initializer = 'ones' if init_weights is None else tf.constant_initializer(init_weights['bn2.weight']),
            moving_mean_initializer = 'zeros' if init_weights is None else tf.constant_initializer(init_weights['bn2.running_mean']),
            moving_variance_initializer = 'ones' if init_weights is None else tf.constant_initializer(init_weights['bn2.running_var'])
            )

        self.downsample = downsample
        self.stride = stride

    def call(self, x, is_training=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, is_training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, is_training)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DownSample(tf.keras.Model):
    def __init__(self, planes, stride, init_weights=None):
        super(DownSample, self).__init__()
        self.convd = tf.keras.layers.Conv2D(
            planes, kernel_size=1,
            strides=stride, use_bias=False, data_format="channels_first",
            kernel_initializer=tf.random_normal_initializer(stddev=0.001) if init_weights is None else tf.constant_initializer(np.transpose(init_weights['0.weight'], (2, 3, 1, 0)))
        )
        self.bnd = tf.keras.layers.BatchNormalization(
            axis=1,
            momentum=BN_MOMENTUM,
            beta_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['1.bias']),
            gamma_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['1.weight']),
            moving_mean_initializer='zeros' if init_weights is None else tf.constant_initializer(init_weights['1.running_mean']),
            moving_variance_initializer='ones' if init_weights is None else tf.constant_initializer(init_weights['1.running_var']))

    def call(self, x, is_training=False):
        x = self.convd(x)
        x = self.bnd(x, is_training)
        return x


class ResNetLayer(tf.keras.Model):
    def __init__(self, inplanes, block, planes, num_blocks, stride=1, init_weights=None):
        super(ResNetLayer, self).__init__()
        self.downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            self.downsample = DownSample(
                planes * block.expansion, stride,
                init_weights=getSpecificWeights(getSpecificWeights(init_weights, "", 0), "downsample", "")
            )

        self.blocks = [block(
            planes,
            stride,
            self.downsample,
            init_weights=getSpecificWeights(init_weights, "", 0))]
        for i in range(1, num_blocks):
            self.blocks.append(
                block(
                    planes,
                    init_weights=getSpecificWeights(init_weights, "", i)
                )
            )

    def call(self, x, is_training=False):

        for block in self.blocks:
            x = block(x, is_training)

        return x


def moveAxis(k, x):
    if x.ndim == 4:
        return np.moveaxis(x, [0, 1, 2, 3], [-1, -2, -4, -3])
    elif x.ndim == 1:
        return x
    else:
        print(k)
        pdb.set_trace()
        raise ValueError()


def getSpecificWeights(weights_dict, name, num):
    if weights_dict is None:
        return None
    else:
        res = {}
        for k, v in weights_dict.items():
            split_k = k.split(".")
            if split_k[0] == "{}{}".format(name, num):
                res[".".join(split_k[1:])] = v
        return res
