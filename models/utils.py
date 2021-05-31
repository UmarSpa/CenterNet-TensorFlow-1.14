import tensorflow as tf

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag_scope, tag, value, global_step):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            with tf.name_scope(tag_scope):
                tf.contrib.summary.scalar(tag, value, step=global_step)

    def log_image(self, tag_scope, tag, value, global_step):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            with tf.name_scope(tag_scope):
                tf.contrib.summary.image(tag, value, step=global_step)


def ckpt_init(model, optimizer, sys_cfg, model_cfg):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, sys_cfg['output_dir'] + '/checkpoints', max_to_keep=25)
    start_iter = 0
    if model_cfg['checkpoint'] is not None:
        ckpt.restore(tf.train.latest_checkpoint(model_cfg['checkpoint']))
        start_iter = int(tf.train.latest_checkpoint(model_cfg['checkpoint']).split('-')[-1])
        print("Ckpt loaded:", tf.train.latest_checkpoint(model_cfg['checkpoint']))
        # CHECK: learning rate is loaded properly, with respect to the start_iter
    else:
        print("No ckpt loaded")

    return ckpt, ckpt_manager, start_iter