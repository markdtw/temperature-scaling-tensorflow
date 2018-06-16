import os
import pdb
import argparse
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from data_utils import CifarDataSet
import resnet_v1

slim = tf.contrib.slim


def get_train_ops(args, num_train_batches, loss_op):

    print ("Model built, getting train ops...")

    # global step
    gstep_op = tf.train.get_or_create_global_step()

    # learning rate decay
    boundaries = [num_train_batches * epoch for epoch in [2, 100, 150, 200]]
    values = [args.init_lr * decay for decay in [1, 10, 0.1, 0.01, 0.001]]
    lr_op = tf.train.piecewise_constant(gstep_op, boundaries, values)

    # optimizer
    optim = tf.train.MomentumOptimizer(lr_op, args.momentum)

    # compute gradient + apply gradient = mimimize
    minimize_op = optim.minimize(loss_op, gstep_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    # tf saver, session
    saver = tf.train.Saver(max_to_keep=2)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            force_gpu_compatible=True,
            allow_growth=True)
    )

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    return gstep_op, lr_op, train_op, saver, sess


def train(args):

    dataset = CifarDataSet(args.batch_size, args.data_dir)
    dataset.make_batch_train()
    dataset.make_batch_valid_or_test()
    if 'cifar-100' in args.data_dir:
        num_classes = 100
    else:
        num_classes = 10

    model = resnet_v1.resnet_v1_110
    # it's actually a 112 since there are 2 additional 1x1 conv for shortcuts
    print ("Data loaded! Building model...")

    # for training
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = model(dataset.images_train, num_classes)
        logits = tf.squeeze(net, [1, 2], name='SqueezedLogits')

    # for evaluating
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net_eval, _ = model(dataset.images_vt, num_classes, is_training=False, reuse=True)
        predictions = tf.argmax(tf.squeeze(net_eval, [1, 2]), axis=-1)

    cross_entropy_loss_op = tf.losses.sparse_softmax_cross_entropy(
        labels=dataset.labels_train, logits=logits)

    l2_loss_op = tf.losses.get_regularization_loss()

    loss_op = cross_entropy_loss_op + l2_loss_op

    num_train_batches = dataset.images_np['train'].shape[0] // args.batch_size
    gstep_op, lr_op, train_op, saver, sess = get_train_ops(args, num_train_batches, loss_op)

    print ("Train ops get! Start training...")

    while True:

        cross_entropy_loss, l2_loss, gstep, lr, _ = sess.run([
            cross_entropy_loss_op, l2_loss_op, gstep_op, lr_op, train_op
        ])

        cur_epoch = gstep // num_train_batches + 1

        if gstep % args.log_every == 0:
            log_string = "({:5d}/{:5d})".format(gstep, num_train_batches * args.epoch)
            log_string += " cross entropy loss: {:.4f}, l2 loss: {:.4f},".format(cross_entropy_loss, l2_loss)
            log_string += " lr: {:.4f}".format(lr)
            log_string += " (ep: {:3d})".format(cur_epoch)
            print (log_string)

        if (gstep + 1) % num_train_batches == 0:

            print ("Saving .ckpt and evaluating with validation set...")

            saver.save(sess, os.path.join(args.save_dir, 'model.ckpt'), global_step=cur_epoch)

            sess.run(dataset.iterator_vt.initializer, feed_dict={dataset.validation: True})

            corrects = 0
            num_eval_batches = dataset.images_np['valid'].shape[0] // dataset.eval_batch_size
            for step in range(num_eval_batches):
                preds, labels = sess.run([predictions, dataset.labels_vt])
                corrects += np.sum(preds == labels)

            print ("validation accuracy: {:.3f}% ({:4d}/{:4d})".format(
                100 * corrects / dataset.images_np['valid'].shape[0],\
                corrects, dataset.images_np['valid'].shape[0]
            ))
            print ("=" * 80)

        if (gstep + 1) % (num_train_batches * args.eval_every) == 0:

            print ("Evaluating with test set...")

            sess.run(dataset.iterator_vt.initializer, feed_dict={dataset.validation: False})

            corrects = 0
            num_eval_batches = dataset.images_np['test'].shape[0] // dataset.eval_batch_size
            for step in range(num_eval_batches):
                preds, labels = sess.run([predictions, dataset.labels_vt])
                corrects += np.sum(preds == labels)

            print ("test accuracy: {:.3f}% ({:5d}/{:5d})".format(
                100 * corrects / dataset.images_np['test'].shape[0],
                corrects, dataset.images_np['test'].shape[0]
            ))
            print ("=" * 80)

        if cur_epoch > args.epoch:
            break

    print ("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr',
        type=float,
        default=1e-2,
        help="initial learning rate.")
    parser.add_argument('--momentum',
        type=float,
        default=9e-1,
        help="Momentum for SGD.")
    parser.add_argument('--epoch',
        type=int,
        default=250,
        help="number of epochs to train.")
    parser.add_argument('--batch-size',
        type=int,
        default=128,
        help="batch size.")
    parser.add_argument('--log-every',
        type=int,
        default=100,
        help="Log every n iterations.")
    parser.add_argument('--eval-every',
        type=int,
        default=5,
        help="Evaluate with test set every m epochs.")
    parser.add_argument('--save-dir',
        type=str,
        default='./log',
        help="Where to save the models.")
    parser.add_argument('--data-dir',
        type=str,
        default='./data/cifar-100-python',
        help="Where the data are saved")
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit("Unknown argument: {}".format(unparsed))
    train(args)
