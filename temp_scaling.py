import os
import pdb
import argparse
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from data_utils import CifarDataSet
import resnet_v1

slim = tf.contrib.slim


def temp_scaling(logits_nps, labels_nps, sess, maxiter=50):

    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))

    logits_tensor = tf.constant(logits_nps, name='logits_valid')
    labels_tensor = tf.constant(labels_nps, name='labels_valid')

    acc_op = tf.metrics.accuracy(labels_tensor, tf.argmax(logits_tensor, axis=1))

    logits_w_temp = tf.divide(logits_tensor, temp_var)

    # loss
    nll_loss_op = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_tensor, logits=logits_w_temp)
    org_nll_loss_op = tf.identity(nll_loss_op)

    # optimizer
    optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': maxiter})

    sess.run(temp_var.initializer)
    sess.run(tf.local_variables_initializer())
    org_nll_loss = sess.run(org_nll_loss_op)

    optim.minimize(sess)

    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)
    acc = sess.run(acc_op)

    print ("Original NLL: {:.3f}, validation accuracy: {:.3f}%".format(org_nll_loss, acc[0] * 100))
    print ("After temperature scaling, NLL: {:.3f}, temperature: {:.3f}".format(
        nll_loss, temperature[0]))

    return temp_var

def main(args):

    dataset = CifarDataSet(args.batch_size, args.data_dir)
    dataset.make_batch_valid_or_test()
    if 'cifar-100' in args.data_dir:
        num_classes = 100
    else:
        num_classes = 10

    model = resnet_v1.resnet_v1_110
    # it's actually a 112 since there are 2 additional 1x1 conv for shortcuts
    print ("Data loaded! Building model...")

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, _ = model(dataset.images_vt, num_classes, is_training=False)
        logits = tf.squeeze(net, [1, 2])

    # tf saver, session
    restorer = tf.train.Saver()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            force_gpu_compatible=True,
            allow_growth=True)
    )

    sess = tf.Session(config=config)
    sess.run(dataset.iterator_vt.initializer, feed_dict={dataset.validation: True})

    restorer.restore(sess, tf.train.latest_checkpoint(args.save_dir))

    print ("Model built! Getting logits...")

    logits_nps = []
    num_eval_batches = dataset.images_np['valid'].shape[0] // dataset.eval_batch_size
    for step in range(num_eval_batches):
        logits_np = sess.run(logits)
        logits_nps.append(logits_np)

    logits_nps = np.concatenate(logits_nps)

    print ("Logits get! Do temperature scaling...")
    print ("=" * 80)

    temp_var = temp_scaling(logits_nps, dataset.labels_np['valid'], sess)
    # use temp_var with your logits to get calibrated output

    print ("=" * 80)
    print ("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',
        type=int,
        default=128,
        help="batch size.")
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
    main(args)
