import os
import time

import numpy as np
import argparse
import tensorflow as tf

from contextlib2 import nullcontext
from config.utils import init_cfg
from db.data_loader import init_db
from models.CenterNet import centernet
from models.CenterNet.results import process_output, compute_result
from models.utils import Tensorboard, ckpt_init
from utils import distributed_train_step, train_step, eval_step, print_losses, print_val

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='./config/config.yaml', help='Path to configuration file')
    return parser.parse_args()


def train():

    # Config
    sys_cfg, db_cfg, model_cfg = init_cfg(parseArgs())

    # Dataset
    train_db, valid_db, val_cfg, valid_db_raw = init_db(db_cfg, sys_cfg)

    # Strategy
    strategy = tf.distribute.MirroredStrategy() if sys_cfg['multi_gpu'] else None

    with strategy.scope() if sys_cfg['multi_gpu'] else nullcontext():

        if sys_cfg['multi_gpu']:
            train_db = strategy.experimental_distribute_dataset(train_db)

        # Model
        model = centernet.CenterNet(model_cfg)

        # Learning rate & Optimizer
        lr = tf.Variable(sys_cfg["learning_rate"])
        optimizer = tf.keras.optimizers.Adam(lr) if sys_cfg['opt_algo'] == 'adam' else tf.keras.optimizers.SGD(lr, momentum=0.9)

        # Checkpoint
        ckpt, ckpt_manager, start_iter = ckpt_init(model, optimizer, sys_cfg, model_cfg)

        # Tensorboard
        tensorboard = Tensorboard(sys_cfg['output_dir'] + '/logs')

        # Training loop
        for (train_iter, inputs) in enumerate(train_db):

            # Initialization
            if (train_iter + start_iter) % sys_cfg['stepsize'] == 0 and (train_iter + start_iter) > start_iter:
                optimizer.lr.assign(optimizer.learning_rate.numpy() / sys_cfg['decay_rate'])

            # Training iter with loss computation
            start = time.time()
            if sys_cfg['multi_gpu']:
                losses = distributed_train_step(inputs=inputs[0:4], outputs=inputs[4:11], model=model, optimizer=optimizer, strategy=strategy, num_GPUS= sys_cfg['NUM_GPUS'])
            else:
                losses = train_step(inputs=inputs[0:4], outputs=inputs[4:11], model=model, optimizer=optimizer)

            # Output log
            if (train_iter + start_iter) % sys_cfg['display'] == 0:
                print_losses(train_iter + start_iter, losses, start)

                tensorboard.log_scalar('training', 'learning_rate', optimizer.learning_rate.numpy(), (train_iter + start_iter))

                for idX, eleX in enumerate(["training", "focal", "pull", "push", "regr"]):
                    tensorboard.log_scalar('training', '{} loss'.format(eleX), losses[idX], (train_iter + start_iter))

            # Validation loop
            if (train_iter + start_iter) % sys_cfg['val_iter'] == 0 and (train_iter + start_iter) > start_iter:
                top_bboxes = {}
                avgTimeIter, avgTimeProc = [], []
                for (test_iter, test_inputs) in enumerate(valid_db):
                    start = time.time()
                    val_out = eval_step(inputs=test_inputs[0], model=model, ae_threshold=val_cfg['ae_threshold'], top_k= val_cfg['top_k'], nms_kernel=val_cfg['nms_kernel'])
                    iter_time = time.time() - start
                    top_bboxes_batch, post_time = process_output(test_inputs[1].numpy(), test_inputs[2].numpy(),
                                                              [val_out[2][0], val_out[2][1]], val_out[0].numpy(),
                                                              val_out[1].numpy(), valid_db_raw,
                                                              test_mod=db_cfg['input_mod'],
                                                              printFlag=test_iter < 32,
                                                              name_base=sys_cfg['output_dir'] + '/val-preds/',
                                                              TB_obj=tensorboard,
                                                              TB_iter=(train_iter + start_iter))

                    avgTimeIter.append(iter_time)
                    avgTimeProc.append(post_time)

                    top_bboxes.update(top_bboxes_batch)
                    print_val(test_iter, np.mean(avgTimeIter[1:]), np.mean(avgTimeProc[1:]))

                compute_result(top_bboxes, valid_db_raw, 80, TB_obj=tensorboard, TB_iter=(train_iter + start_iter), out_dir=sys_cfg['output_dir'])

            # Save checkpoint
            if (train_iter + start_iter) % sys_cfg['snapshot'] == 0 and (train_iter + start_iter) > start_iter:
                ckpt_manager.save(checkpoint_number=(train_iter + start_iter))

        tensorboard.close()


if __name__ == "__main__":
    train()
