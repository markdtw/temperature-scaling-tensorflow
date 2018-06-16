"""Modified to train on CIFAR-10/100 by Mark

Typical use:

   from tensorflow.contrib.slim.python.slim.nets import
   resnet_v1

ResNet-110 for image classification into 10 classes:

   # inputs has shape [batch, 32, 32, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_110(inputs, 10, is_training=False)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

resnet_arg_scope = resnet_utils.resnet_arg_scope


@add_arg_scope
def not_bottleneck(inputs,
               depth,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  with variable_scope.variable_scope(scope, 'not_bottleneck_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = layers.conv2d(
        inputs, depth, 3, stride, scope='conv1')
    residual = layers.conv2d(
        residual, depth, 3, stride=1, activation_fn=None, scope='conv2')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  with variable_scope.variable_scope(
      scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers.conv2d, not_bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      with arg_scope([layers.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = resnet_utils.conv2d_same(net, 16, 3, stride=1, scope='conv1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        if global_pool:
          # Global average pooling.
          net = math_ops.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
        if num_classes is not None:
          net = layers.conv2d(
              net,
              num_classes, [1, 1],
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = layers_lib.softmax(
              net, scope='predictions')
        return net, end_points
resnet_v1.default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride):
  return resnet_utils.Block(scope, not_bottleneck, [{
      'depth': base_depth,
      'stride': stride
  }] + (num_units - 1) * [{
      'depth': base_depth,
      'stride': 1
  }])


def resnet_v1_110(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v1_110'):
  blocks = [
      resnet_v1_block('block1', base_depth=16, num_units=18, stride=1),
      resnet_v1_block('block2', base_depth=32, num_units=18, stride=2),
      resnet_v1_block('block3', base_depth=64, num_units=18, stride=2),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)

