import time
import tensorflow as tf
from models.CenterNet.loss import centernet_loss

def train_step_dist(inputs, outputs, model, optimizer, num_GPUS):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        loss, L1, L2, L3, L4 = centernet_loss(outputs, preds)
        loss, L1, L2, L3, L4 = scale_losses([loss, L1, L2, L3, L4], num_GPUS)
    grads = tape.gradient(loss, model.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 10.)
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    return loss, L1, L2, L3, L4


@tf.function
def distributed_train_step(inputs, outputs, model, optimizer, strategy, num_GPUS):
    per_replica_loss, per_replica_L1, per_replica_L2, per_replica_L3, per_replica_L4 = strategy.experimental_run_v2(
        train_step_dist,
        args=(inputs, outputs, model, optimizer, num_GPUS))

    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None), strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_L1, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                                per_replica_L2,
                                                                                axis=None), strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_L3, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                                per_replica_L4, axis=None)


# @tf.function
def train_step(inputs, outputs, model, optimizer):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        loss, L1, L2, L3, L4 = centernet_loss(outputs, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 10.)
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    return loss, L1, L2, L3, L4


@tf.function
def eval_step(inputs, model, ae_threshold, top_k, nms_kernel):
    pred_bbs, pred_cnts, ht_shape = model([inputs, [], [], []], training=False, ae_threshold=ae_threshold, K=top_k,
                                          kernel=nms_kernel)
    return pred_bbs, pred_cnts, ht_shape

def print_losses(iter, losses, start):
    print("iteration {0:7d}: total loss - {1:7.4f}, focal - {2:7.4f}, pull - {3:6.4f}, push - {4:6.4f}, regr - {5:6.4f}, time/iter - {6:6.4f} sec".format(
        iter, losses[0], losses[1], losses[2], losses[3], losses[4], time.time() - start))


def print_val(test_iter, time_iter, time_pro):
    print(
    "Val iteration {0:4d}, average iteration time {1:7.5f} sec, average post processing time {2:7.5f} sec".format(
        test_iter, time_iter, time_pro))


def scale_losses(losses, num_GPUS):
    return [loss / num_GPUS for loss in losses]
