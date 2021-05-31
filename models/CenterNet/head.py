import tensorflow as tf
from .utils import conv_block, kp_layer
from .pooling import tl_pool, br_pool, ct_pool


class Head(tf.keras.Model):
    def __init__(self, weights_dic=None):
        super(Head, self).__init__()
        self.cnv = conv_block(out_dim=256, kernel=3, stride=1, padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                              weights_init=None if weights_dic is None else weights_dic['cnv'])

        self.tl_cnv = tl_pool(weights_dic=None if weights_dic is None else weights_dic['tl_cnv'])
        self.br_cnv = br_pool(weights_dic=None if weights_dic is None else weights_dic['br_cnv'])
        self.ct_cnv = ct_pool(weights_dic=None if weights_dic is None else weights_dic['ct_cnv'])

        self.tl_heat = kp_layer(outdim=80, weights_dic=None if weights_dic is None else weights_dic['tl_heat'])
        self.br_heat = kp_layer(outdim=80, weights_dic=None if weights_dic is None else weights_dic['br_heat'])
        self.ct_heat = kp_layer(outdim=80, weights_dic=None if weights_dic is None else weights_dic['ct_heat'])

        self.tl_tag = kp_layer(outdim=1, weights_dic=None if weights_dic is None else weights_dic['tl_tag'])
        self.br_tag = kp_layer(outdim=1, weights_dic=None if weights_dic is None else weights_dic['br_tag'])

        self.tl_regr = kp_layer(outdim=2, weights_dic=None if weights_dic is None else weights_dic['tl_regr'])
        self.br_regr = kp_layer(outdim=2, weights_dic=None if weights_dic is None else weights_dic['br_regr'])
        self.ct_regr = kp_layer(outdim=2, weights_dic=None if weights_dic is None else weights_dic['ct_regr'])

    def call(self, inputs, training=None, mask=None, decode=None, ae_threshold=None, K=None, kernel=None):

        backbone, tl_ind, br_ind, ct_ind = inputs

        cnv = self.cnv(backbone, training=training)
        tl_cnv = self.tl_cnv(cnv, training=training)
        br_cnv = self.br_cnv(cnv, training=training)
        ct_cnv = self.ct_cnv(cnv, training=training)

        tl_heat = self.tl_heat(tl_cnv)
        br_heat = self.br_heat(br_cnv)
        ct_heat = self.ct_heat(ct_cnv)

        tl_tag = self.tl_tag(tl_cnv)
        br_tag = self.br_tag(br_cnv)

        tl_regr = self.tl_regr(tl_cnv)
        br_regr = self.br_regr(br_cnv)
        ct_regr = self.ct_regr(ct_cnv)

        if training:
            tl_tag = self._tranpose_and_gather_feat(tl_tag, tl_ind)
            br_tag = self._tranpose_and_gather_feat(br_tag, br_ind)
            tl_regr = self._tranpose_and_gather_feat(tl_regr, tl_ind)
            br_regr = self._tranpose_and_gather_feat(br_regr, br_ind)
            ct_regr = self._tranpose_and_gather_feat(ct_regr, ct_ind)
            return [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr, ct_regr]
        else:
            return self._decode(tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, K=K,
                                kernel=kernel, ae_threshold=ae_threshold)

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = tf.transpose(feat, [0, 2, 3, 1])
        feat = tf.reshape(feat, [tf.shape(feat)[0], -1, tf.shape(feat)[3]])
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        feat = tf.batch_gather(feat, ind)
        # TODO: implement mask
        return feat

    def _nms(self, heat, kernel=1):
        hmax = tf.nn.max_pool(heat, ksize=kernel, strides=(1, 1), data_format="NCHW", padding='SAME')
        keep = tf.cast(tf.equal(hmax, heat), tf.float32)
        return heat * keep

    def _topk(self, scores, K=20):
        batch, cat, height, width = scores.get_shape().as_list()
        heat = tf.reshape(scores, (batch, -1))
        topk_scores, topk_inds = tf.nn.top_k(heat, K)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = tf.cast(topk_inds // width, tf.float32)
        topk_xs = tf.cast(topk_inds % width, tf.float32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def _decode(self, tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, K=100, kernel=1,
                ae_threshold=1, num_dets=1000):
        batch, cat, height, width = tl_heat.get_shape().as_list()

        tl_heat = tf.nn.sigmoid(tl_heat)
        br_heat = tf.nn.sigmoid(br_heat)
        ct_heat = tf.nn.sigmoid(ct_heat)

        tl_heat = self._nms(tl_heat, kernel=kernel)
        br_heat = self._nms(br_heat, kernel=kernel)
        ct_heat = self._nms(ct_heat, kernel=kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, K=K)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, K=K)
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_heat, K=K)

        tl_ys = tf.tile(tf.expand_dims(tl_ys, axis=2), [1, 1, K])
        tl_xs = tf.tile(tf.expand_dims(tl_xs, axis=2), [1, 1, K])
        br_ys = tf.tile(tf.expand_dims(br_ys, axis=1), [1, K, 1])
        br_xs = tf.tile(tf.expand_dims(br_xs, axis=1), [1, K, 1])
        ct_ys = tf.tile(tf.expand_dims(ct_ys, axis=1), [1, K, 1])
        ct_xs = tf.tile(tf.expand_dims(ct_xs, axis=1), [1, K, 1])

        if tl_regr is not None and br_regr is not None:
            tl_regr = self._tranpose_and_gather_feat(tl_regr, tl_inds)
            tl_regr = tf.expand_dims(tl_regr, axis=2)
            br_regr = self._tranpose_and_gather_feat(br_regr, br_inds)
            br_regr = tf.expand_dims(br_regr, axis=1)
            ct_regr = self._tranpose_and_gather_feat(ct_regr, ct_inds)
            ct_regr = tf.expand_dims(ct_regr, axis=1)

            tl_xs = tl_xs + tl_regr[..., 0]
            tl_ys = tl_ys + tl_regr[..., 1]
            br_xs = br_xs + br_regr[..., 0]
            br_ys = br_ys + br_regr[..., 1]
            ct_xs = ct_xs + ct_regr[..., 0]
            ct_ys = ct_ys + ct_regr[..., 1]

        # all possible boxes based on top k corners (ignoring class)
        bboxes = tf.stack((tl_xs, tl_ys, br_xs, br_ys), axis=-1)

        tl_tag = self._tranpose_and_gather_feat(tl_tag, tl_inds)
        tl_tag = tf.expand_dims(tl_tag, axis=2)
        br_tag = self._tranpose_and_gather_feat(br_tag, br_inds)
        br_tag = tf.expand_dims(br_tag, axis=1)

        dists = tf.squeeze(tf.abs(tl_tag - br_tag), axis=3)

        tl_scores = tf.tile(tf.expand_dims(tl_scores, axis=2), [1, 1, K])
        br_scores = tf.tile(tf.expand_dims(br_scores, axis=1), [1, K, 1])
        scores = (tl_scores + br_scores) / 2

        tl_clses = tf.tile(tf.expand_dims(tl_clses, axis=2), [1, 1, K])
        br_clses = tf.tile(tf.expand_dims(br_clses, axis=1), [1, K, 1])

        rej_mask = -tf.ones_like(scores)

        scores = tf.where(tf.not_equal(tl_clses, br_clses), rej_mask, scores)
        scores = tf.where(tf.greater(dists, ae_threshold), rej_mask, scores)
        scores = tf.where(tf.less(br_xs, tl_xs), rej_mask, scores)
        scores = tf.where(tf.less(br_ys, tl_ys), rej_mask, scores)
        scores = tf.reshape(scores, (batch, -1))
        scores, inds = tf.nn.top_k(scores, num_dets)
        scores = tf.expand_dims(scores, -1)

        bboxes = tf.reshape(bboxes, (batch, -1, 4))
        bboxes = self._gather_feat(bboxes, inds)

        clses = tf.reshape(tl_clses, (batch, -1, 1))
        clses = tf.cast(self._gather_feat(clses, inds), tf.float32)

        tl_scores = tf.reshape(tl_scores, (batch, -1, 1))
        tl_scores = tf.cast(self._gather_feat(tl_scores, inds), tf.float32)

        br_scores = tf.reshape(br_scores, (batch, -1, 1))
        br_scores = tf.cast(self._gather_feat(br_scores, inds), tf.float32)

        ct_xs = ct_xs[:, 0, :]
        ct_ys = ct_ys[:, 0, :]

        center = tf.concat(
            [tf.expand_dims(ct_xs, 2), tf.expand_dims(ct_ys, 2), tf.expand_dims(tf.cast(ct_clses, tf.float32), 2),
             tf.expand_dims(ct_scores, 2)], -1)
        detections = tf.concat([bboxes, scores, tl_scores, br_scores, clses], -1)
        return detections, center, [height, width]
