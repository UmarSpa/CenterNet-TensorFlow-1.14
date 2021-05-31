import tensorflow as tf
from .Hourglass import Pre
from .Hourglass import HourGlass52
from .Resnet import ResNet50, ResNet18
from .Resnet import Post_ResNet
from .head import Head
from .loss import centernet_loss
from .pretrained_weights_PT.weights_mapping import pre_weights, backbone_weights, head_weights


class Network(tf.keras.Model):
    def __init__(self, cfg_model):
        super(Network, self).__init__()
        self.model = CenterNet(cfg_model)
        self.loss = centernet_loss

    def call(self, inputs, training=None, mask=None):
        xs, ys = inputs["xs"], inputs["ys"]
        preds = self.model(xs, training=training)
        loss = self.loss(ys, preds)
        return loss


class CenterNet(tf.keras.Model):
    def __init__(self, cfg_model):
        super(CenterNet, self).__init__()
        self._cfg(cfg_model)

        self._pre_backbone()
        self._backbone()
        self._post_backbone()
        self._head()

    def call(self, inputs, training=None, mask=None, decode=None, ae_threshold=None, K=None, kernel=None):
        image, x1, x2, x3 = inputs
        pre = self.pre(image, training=training) if self.cfg_model['backbone_type'] == 'hourglass52' else image
        backbone = self.backbone(pre, training=training)
        post = self.post(backbone, training=training) if self.cfg_model['backbone_type'] == 'resnet50' or self.cfg_model['backbone_type'] == 'resnet18' else backbone
        head = self.head([post, x1, x2, x3], training=training, decode=decode, ae_threshold=ae_threshold, K=K,
                         kernel=kernel)
        return head

    def _cfg(self, cfg_model):
        self.cfg_model = cfg_model
        self._load_weights()

    def _pre_backbone(self):
        if self.cfg_model['backbone_type'] == 'hourglass52':
            self.pre = Pre(weights_dic=self.pre_weights)
        elif self.cfg_model['backbone_type'] == 'resnet50' or self.cfg_model['backbone_type'] == 'resnet18':
            pass
        else:
            raise NameError('myError: backbone name not as expected')
        self.pre_weights = None

    def _backbone(self):
        if self.cfg_model['backbone_type'] == 'hourglass52':
            self.backbone = HourGlass52(weights_dic=self.backbone_weights)
        elif self.cfg_model['backbone_type'] == 'resnet50':
            self.backbone = ResNet50(weights_fp=self.backbone_weights)
        elif self.cfg_model['backbone_type'] == 'resnet18':
            self.backbone = ResNet18(weights_fp=self.backbone_weights)
        else:
            raise NameError('myError: backbone name not as expected')
        self.hourglass_weights = None

    def _post_backbone(self):
        if self.cfg_model['backbone_type'] == 'hourglass52':
            pass
        elif self.cfg_model['backbone_type'] == 'resnet50':
            self.post = Post_ResNet(backbone='R-50', weight_path=self.post_weights)
        elif self.cfg_model['backbone_type'] == 'resnet18':
            self.post = Post_ResNet(backbone='R-18', weight_path=self.post_weights)
        else:
            raise NameError('myError: backbone name not as expected.')
        self.pre_weights = None

    def _head(self):
        self.head = Head(weights_dic=self.head_weights)
        self.head_weights = None
        pass

    def _load_weights(self):
        if self.cfg_model['backbone_type'] == 'hourglass52':
            self.pre_weights = pre_weights(self.cfg_model['pre']['init_HG_PT'])
            self.backbone_weights = backbone_weights(self.cfg_model['backbone']['init_HG_PT'])
        elif self.cfg_model['backbone_type'] == 'resnet50':
            self.backbone_weights = self.cfg_model['backbone']['init_resnet']
            self.post_weights = self.cfg_model['post']['init_resnet']
        elif self.cfg_model['backbone_type'] == 'resnet18':
            self.backbone_weights = self.cfg_model['backbone']['init_resnet']
            self.post_weights = self.cfg_model['post']['init_resnet']
        self.head_weights = head_weights(self.cfg_model['head']['init_PT'])