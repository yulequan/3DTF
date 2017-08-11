import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

#######################
# 3d functions
#######################
# convolution
def conv3d(input, output_chn, kernel_size, stride=1, data_format='channels_last', use_bias=False, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format=data_format,
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            use_bias=use_bias, name=name)


def conv3d_bn_relu(input, output_chn, kernel_size, stride, channel_dim =-1, use_bias=False, training=True, name='conv'):
    if channel_dim==1:
        data_format = 'channels_first'
    else:
        data_format ='channels_last'
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride, data_format, use_bias, name='conv')
        bn = tf.layers.batch_normalization(conv, momentum=0.9, axis=channel_dim, training=training, name='bn')
        relu = tf.nn.relu(bn, name='relu')
    return relu


# deconvolution
def deconv3d(input, output_chn, kernel_size = 4, stride = 2, data_format='channels_last', use_bias = False,
             use_bilinear=False,name='deconv'):
    axis = 4 if data_format == "channels_last" else 1
    in_chn = input.get_shape()[axis]

    if use_bilinear:
        init = get_bilinear_initializer(output_chn,in_chn)
    else:
        init = tf.truncated_normal_initializer(0.0, 0.01)

    return tf.layers.conv3d_transpose(inputs=input,filters=output_chn,kernel_size=kernel_size,strides=stride,
                                      padding='same',data_format=data_format, kernel_initializer=init,
                                      use_bias=use_bias, name=name)


def deconv3d_bn_relu(input, output_chn, kernel_size = 4, stride = 2, channel_dim =-1, use_bias = False, training = True, name='deconv'):
    if channel_dim==1:
        data_format = 'channels_first'
    else:
        data_format ='channels_last'
    with tf.variable_scope(name):
        deconv = deconv3d(input, output_chn,kernel_size,stride,data_format,use_bias,name='deconv')
        bn = tf.layers.batch_normalization(deconv, momentum=0.9,axis=channel_dim, training=training, name='bn')
        relu = tf.nn.relu(bn, name='relu')
    return relu


def get_bilinear_initializer(in_chn, out_chn, filter_shape=4, upscale_factor=2):
    ##filter_shape is [width, height, lenth, num_out_channels,num_in_channels, ]
    kernel_size = filter_shape

    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape, filter_shape, filter_shape])
    for x in range(filter_shape):
        for y in range(filter_shape):
            for z in range(filter_shape):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * \
                        (1 - abs((y - centre_location) / upscale_factor))*\
                        (1 - abs((z - centre_location) / upscale_factor))
                bilinear[x, y,z] = value
    weights = np.zeros((filter_shape,filter_shape,filter_shape,out_chn,in_chn))
    for i in range(out_chn):
        weights[:, :, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    return init


def conv_block(x, stage, branch, num_filter, concat_axis=4, dropout_rate=None, training=True):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
    '''
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)
    if concat_axis == 1:
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    # 3x3 Convolution
    x = tf.layers.batch_normalization(x, axis=concat_axis,momentum=0.9,training=training, name=conv_name_base + '_x2_bn')
    x = tf.nn.relu(x, name=relu_name_base + '_x2')
    x = tf.layers.conv3d(x, num_filter, [3, 3, 3], padding='same', data_format=data_format,
                         kernel_initializer=tf.random_normal_initializer(0.0, 0.01), use_bias=False,
                         name=conv_name_base + '_x2')
    if dropout_rate:
        x = tf.layers.dropout(x, dropout_rate, training=training)

    return x


def dense_block(x, stage, num_layers, num_filter, growth_rate, concat_axis=-1, dropout_rate=None, training=True, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''
    concat_feat = x
    for i in range(num_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, concat_axis,dropout_rate,training)
        concat_feat = tf.concat([concat_feat,x],axis=concat_axis,name='concat_'+str(stage)+'_'+str(branch))
        if grow_nb_filters:
            num_filter += growth_rate
    return concat_feat, num_filter


def transition_block(x, stage, num_filter, compression=1.0, concat_axis=-1, dropout_rate=None, training=True):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    # pool_name_base = 'pool' + str(stage)
    if concat_axis == 1:
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    x = tf.layers.batch_normalization(x, axis=concat_axis, momentum=0.9, training=training, name=conv_name_base + '_bn')
    x = tf.nn.relu(x, name=relu_name_base)
    x = tf.layers.conv3d(x, int(num_filter * compression), [1, 1, 1], padding='same', data_format=data_format,
                         kernel_initializer=tf.random_normal_initializer(0.0, 0.01), use_bias=False,
                         name=conv_name_base)
    # if dropout_rate:
    #     x_drop = tf.layers.dropout(x, dropout_rate, training=training)

    return x,int(num_filter * compression)




# dice loss function
def dice_loss_fun(self, pred, input_gt):
    input_gt = tf.one_hot(input_gt, 8)
    # print(input_gt.shape)
    dice = 0
    for i in range(8):
        inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
        l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
        r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
        dice = dice + 2*inse/(l+r)
    return -dice

def softmax_weighted_loss(labels, logits, weights=1.0, num_classes= None,loss_collection=tf.GraphKeys.LOSSES,scope=None):
    with tf.name_scope(scope):
        if num_classes is None:
            num_classes = logits.get_shape()[-1]
        labels = tf.one_hot(labels, depth=num_classes)
        epsilon = tf.constant(value=1e-10)
        softmax = tf.nn.softmax(logits+epsilon)+epsilon
        weighted = 1.0-tf.to_float(tf.reduce_sum(labels,axis=[0,1,2,3]))/tf.to_float(tf.reduce_sum(labels))
        cross_entropy = -tf.reduce_sum(labels*tf.log(softmax)*weighted,axis=-1)
        cross_entropy_mean = weights*tf.reduce_mean(cross_entropy, name='xentropy')
        tf.add_to_collection(loss_collection, cross_entropy_mean)
    return cross_entropy_mean