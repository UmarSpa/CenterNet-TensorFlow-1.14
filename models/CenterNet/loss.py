import tensorflow as tf
from .utils import sigmoid


def centernet_loss(y_true, y_pred):
    pull_weight = 0.1
    push_weight = 0.1
    regr_weight = 1

    tl_heat = y_pred[0]
    br_heat = y_pred[1]
    ct_heat = y_pred[2]
    tl_tag = y_pred[3]
    br_tag = y_pred[4]
    tl_regr = y_pred[5]
    br_regr = y_pred[6]
    ct_regr = y_pred[7]

    gt_tl_heat = y_true[0]
    gt_br_heat = y_true[1]
    gt_ct_heat = y_true[2]
    gt_mask = y_true[3]
    gt_tl_regr = y_true[4]
    gt_br_regr = y_true[5]
    gt_ct_regr = y_true[6]

    # focal loss
    focal_loss = 0
    tl_heat = sigmoid(tl_heat)
    br_heat = sigmoid(br_heat)
    ct_heat = sigmoid(ct_heat)
    focal_loss += focal_loss_fn(tl_heat, gt_tl_heat)
    focal_loss += focal_loss_fn(br_heat, gt_br_heat)
    focal_loss += focal_loss_fn(ct_heat, gt_ct_heat)

    # tag loss
    pull_loss, push_loss = tag_loss_fn(tl_tag, br_tag, gt_mask)
    pull_loss = pull_weight * pull_loss
    push_loss = push_weight * push_loss

    # regr loss
    regr_loss = 0
    regr_loss += regr_loss_fn(tl_regr, gt_tl_regr, gt_mask)
    regr_loss += regr_loss_fn(br_regr, gt_br_regr, gt_mask)
    regr_loss += regr_loss_fn(ct_regr, gt_ct_regr, gt_mask)
    regr_loss = regr_weight * regr_loss

    loss = focal_loss + pull_loss + push_loss + regr_loss
    return loss, focal_loss, pull_loss, push_loss, regr_loss


def focal_loss_fn(pred, gt_heat):
    num_pos = tf.reduce_sum(tf.cast(tf.equal(gt_heat, 1), tf.float32))
    neg_weights = tf.pow((1 - tf.boolean_mask(gt_heat, tf.less(gt_heat, 1))), 4)

    pos_pred = tf.boolean_mask(pred, tf.equal(gt_heat, 1))
    neg_pred = tf.boolean_mask(pred, tf.less(gt_heat, 1))

    pos_loss = tf.reduce_sum(tf.log(pos_pred) * tf.pow(1 - pos_pred, 2))
    neg_loss = tf.reduce_sum(tf.log((1 - neg_pred)) * tf.pow(neg_pred, 2) * neg_weights)

    if tf.size(pos_pred) == 0:
        loss = - neg_loss
    else:
        loss = - (pos_loss + neg_loss) / num_pos
    return loss


def tag_loss_fn(tag0, tag1, mask):
    num = tf.cast(tf.reduce_sum(mask, axis=1, keepdims=True), tf.float32)
    tag0 = tf.squeeze(tag0, axis=2)
    tag1 = tf.squeeze(tag1, axis=2)

    tag_mean = (tag0 + tag1) / 2

    tag0 = tf.div(tf.pow(tag0 - tag_mean, 2), (num + 1e-4))
    tag0 = tf.reduce_sum(tf.boolean_mask(tag0, mask))
    tag1 = tf.div(tf.pow(tag1 - tag_mean, 2), (num + 1e-4))
    tag1 = tf.reduce_sum(tf.boolean_mask(tag1, mask))
    pull = tag0 + tag1

    mask = tf.expand_dims(mask, axis=1) + tf.expand_dims(mask, axis=2)
    mask = tf.equal(mask, 2)
    num = tf.expand_dims(num, axis=2)
    num2 = (num - 1) * num
    dist = tf.expand_dims(tag_mean, axis=1) - tf.expand_dims(tag_mean, axis=2)
    dist = 1 - tf.abs(dist)
    dist = tf.nn.relu(dist)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = tf.boolean_mask(dist, mask)
    push = tf.reduce_sum(dist)
    return pull, push


def regr_loss_fn(regr, gt_regr, mask):
    num = tf.cast(tf.reduce_sum(mask), tf.float32)
    mask = tf.stack((mask, mask), -1)
    regr1 = tf.boolean_mask(regr, mask)
    gt_regr1 = tf.boolean_mask(gt_regr, mask)
    regr_loss = smooth_l1_loss(gt_regr1, regr1)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def smooth_l1_loss(labels, predictions, delta=1.0):
    predictions = tf.cast(predictions, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    abs_error = tf.abs(tf.subtract(predictions, labels))
    loss = tf.where(tf.less(abs_error, delta),
                    tf.multiply(tf.multiply(abs_error, abs_error), 0.5),
                    tf.subtract(abs_error, 0.5))
    return tf.reduce_sum(loss)
