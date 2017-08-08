import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

#######################
# 3d functions
#######################
# convolution
def conv3d(input, output_chn, kernel_size, stride, data_format='channels_last', use_bias=False, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format=data_format,
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            use_bias=use_bias, name=name)


def conv3d_bn_relu(input, output_chn, kernel_size, stride, channel_dim =4, use_bias=False, training=True, name='conv'):
    if channel_dim==1:
        data_format = 'channels_first'
    else:
        data_format ='channels_last'
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride, data_format, use_bias, name='conv')
        bn = tf.layers.batch_normalization(conv, axis=channel_dim, training=training, name='bn')
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


def deconv3d_bn_relu(input, output_chn, kernel_size = 4, stride = 2, channel_dim =4, use_bias = False, training = True, name='deconv'):
    if channel_dim==1:
        data_format = 'channels_first'
    else:
        data_format ='channels_last'
    with tf.variable_scope(name):
        deconv = deconv3d(input, output_chn,kernel_size,stride,data_format,use_bias,name='deconv')
        bn = tf.layers.batch_normalization(deconv, axis=channel_dim, training=training, name='bn')
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

# class-weighted cross-entropy loss function
def softmax_weighted_loss(self, logits, labels):
    """
    Loss = weighted * -target*log(softmax(logits))
    :param logits: probability score
    :param labels: ground_truth
    :return: softmax-weifhted loss
    """
    gt = tf.one_hot(labels, 4)
    pred = logits
    softmaxpred = tf.nn.softmax(pred)
    loss = 0
    for i in range(4):
        gti = gt[:,:,:,:,i]
        predi = softmaxpred[:,:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
        # print("class %d"%(i))
        # print(weighted)
        loss = loss+ -tf.reduce_mean(weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)))
    return loss
