from __future__ import division
from tqdm import tqdm
from ops import *
from utils import *
from seg_eval import *

class Model(object):
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_size    = param_set['inputI_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.output_chn     = param_set['output_chn']
        self.resize_r       = param_set['resize_r']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.model_path     = param_set['model_path']
        self.lr             = param_set['learning_rate']
        self.iter_nums      = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']
        self.ovlp_ita       = param_set['ovlp_ita']
        self.reset_h5       = param_set['reset_h5']
        self.data_format    = param_set['data_format']
        self.up_feat        = param_set['up_feat']

        # build model graph
        self.build_network()

        # trainable variables
        self.u_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    # build model graph
    def build_network(self):
        self.phase = (self.phase == 'train')
        if self.data_format=='channels_last':
            self.chn_dim = 4
            self.input_I = tf.placeholder(dtype=tf.float32, shape=[None, self.inputI_size, self.inputI_size,
                                                                   self.inputI_size, self.inputI_chn], name='inputI')
        else:
            self.chn_dim = 1
            self.input_I = tf.placeholder(dtype=tf.float32, shape=[None, self.inputI_chn, self.inputI_size,
                                                                   self.inputI_size, self.inputI_size], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int32, shape=[None, self.inputI_size, self.inputI_size,
                                                         self.inputI_size], name='target')


        logits = self.UNET_modality_pool5(chn_num=64,fusion_method='concat1')
        #logits = self.UNET_modality32('concat1')
        #logits = self.UNET()
        #main_logits, aux0_logits = self.DenseNet()

        # predicted labels
        self.pred_prob = tf.nn.softmax(logits[0], name='pred_soft')
        self.pred_label = tf.argmax(self.pred_prob, axis=4, name='argmax')

        if not self.phase:
            return

        # ========= calculate loss========
        self.losses = []
        for logit in logits:
            self.losses.append(tf.losses.sparse_softmax_cross_entropy(self.input_gt,logit))
            #self.losses.append(weighted_softmax_cross_entropy(self.input_gt, logit,weighted='median_freq'))
            #self.losses.append(focal_loss(self.input_gt, logit))
            #self.losses.append(dice_loss(self.input_gt,logit))


        # self.aux0_loss = focal_loss(self.input_gt, aux0_logits)
        # self.aux1_loss = focal_loss(self.input_gt, aux1_logits)
        # self.main_loss = focal_loss(self.input_gt, main_logits)

        # apply regularization
        tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.0001),tf.trainable_variables())

        # total loss
        # fused_loss, main_loss, aux2_loss, aux1_loss, aux0_loss
        # weights = [0.3, 0.3, 0.2, 0.1, 0.1]

        # main_loss, aux2_loss, aux1_loss, aux0_loss
        weights = [0.6, 0.2, 0.1, 0.1]

        # main_loss, aux1_loss, aux0_loss
        #weights = [0.625, 0.25, 0.125]

        self.entroy_loss = 0
        for i in range(len(self.losses)):
            self.entroy_loss += weights[i]*self.losses[i]
        self.total_loss = self.entroy_loss + tf.losses.get_regularization_loss()

        # make train op
        self.learning_rate = tf.train.polynomial_decay(self.lr,tf.train.get_or_create_global_step(),
                                                       decay_steps=self.iter_nums, power=0.9)
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.train_op = tf.contrib.training.create_train_op(self.total_loss, optimizer,colocate_gradients_with_ops=True)
        tf.summary.scalar('lr/learning_rate', self.learning_rate)

        # add summary images
        flair = tf.squeeze(tf.slice(self.input_I, [0, 0, 0, 40, 3], [-1, -1, -1, 1, 1]),axis=-1)
        pred = tf.cast(tf.slice(self.pred_label,[0,0,0,40],[-1,-1,-1,1])*80,tf.uint8)
        gt = tf.cast(tf.slice(self.input_gt,[0,0,0,40],[-1,-1,-1,1])*80,tf.uint8)
        tf.summary.image('img/input_flair',flair,max_outputs=3)
        tf.summary.image('img/pred', pred, max_outputs=3)
        tf.summary.image('img/gt', gt, max_outputs=3)

        for i,loss in enumerate(self.losses):
            tf.summary.scalar('losses/entroy_loss_{}'.format(i), loss)
        tf.summary.scalar('losses/entroy_loss',self.entroy_loss)
        tf.summary.scalar('losses/regu_loss', tf.losses.get_regularization_loss())
        tf.summary.scalar('losses/total_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.model_path, 'logs'), self.sess.graph)

    def multimodal_fusion(self,feats, output_chn, method='concat1',name=None):
        with tf.variable_scope(name):
            if method == 'concat1':
                atten_feats = tf.concat(feats, axis=self.chn_dim)

                return conv3d_bn_relu(atten_feats, output_chn, 1, 1, self.chn_dim, training=self.phase, name='conv')
            elif method == 'concat2':
                atten_feats = tf.concat(feats, axis=self.chn_dim)
                pool = tf.layers.average_pooling3d(atten_feats,atten_feats.shape.as_list()[1],
                                                   atten_feats.shape.as_list()[1],data_format=self.data_format,name='global')
                fc = tf.nn.sigmoid(tf.layers.dense(pool,output_chn,name='fc'))
                return conv3d_bn_relu(atten_feats*fc, output_chn, 1, 1, self.chn_dim, training=self.phase, name='conv')


            elif method=='attention1':
                inputs = tf.concat(feats, axis=self.chn_dim)
                attention1 = conv3d(inputs,feats[0].shape[-1],1,1,self.data_format,name='attention1')
                attention2 = conv3d(attention1,4,3,1,self.data_format,name='attention2')
                soft = tf.nn.softmax(attention2,name='soft_attention')
                soft_unstack = tf.unstack(soft,axis=-1)
                atten_feats = tf.concat([tf.expand_dims(w,-1)*feat for w,feat in zip(soft_unstack,feats)], axis=self.chn_dim)

                return conv3d_bn_relu(atten_feats, output_chn, 1, 1, self.chn_dim, training=self.phase, name='conv')
            elif method == 'attention2':
                return 0

    def UNET_modality32(self, fusion_method='concat1'):
        # downsample path
        conv1a = [];pool1 = [];conv2a = [];pool2 = [];conv3a = [];conv3b = [];pool3 = [];conv4a = [];conv4b = []
        for m in range(4):
            with tf.variable_scope('M{}'.format(m)):
                conv1a.append(
                    conv3d_bn_relu(self.input_I[:, :, :, :, m:m + 1], 32, 3, 1, self.chn_dim, training=self.phase,
                                   name='conv1a'))
                pool1.append(tf.layers.max_pooling3d(conv1a[m], 2, 2, data_format=self.data_format, name='pool1'))

                with tf.device("/gpu:1"):
                    conv2a.append(conv3d_bn_relu(pool1[m], 64, 3, 1, self.chn_dim, training=self.phase, name='conv2a'))
                    pool2.append(tf.layers.max_pooling3d(conv2a[m], 2, 2, data_format=self.data_format, name='pool2'))

                    conv3a.append(conv3d_bn_relu(pool2[m], 128, 3, 1, self.chn_dim, training=self.phase, name='conv3a'))
                    conv3b.append(
                        conv3d_bn_relu(conv3a[m], 128, 3, 1, self.chn_dim, training=self.phase, name='conv3b'))
                    pool3.append(tf.layers.max_pooling3d(conv3b[m], 2, 2, data_format=self.data_format, name='pool3'))

                    conv4a.append(conv3d_bn_relu(pool3[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv4a'))
                    conv4b.append(
                        conv3d_bn_relu(conv4a[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv4b'))

        # upsample path
        with tf.device("/gpu:1"):
            conv4 = self.multimodal_fusion(conv4b, 32, fusion_method, name='conv4_fusion')
            conv3 = self.multimodal_fusion(conv3b, 32, fusion_method, name='conv3_fusion')
            conv2 = self.multimodal_fusion(conv2a, 32, fusion_method, name='conv2_fusion')
        conv1 = self.multimodal_fusion(conv1a, 32, fusion_method, name='conv1_fusion')
        with tf.device("/gpu:1"):
            deconv1a = deconv3d_bn_relu(input=conv4, output_chn=32, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv3], axis=self.chn_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=32, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv2a')
            concat2 = tf.concat([deconv2a, conv2], axis=self.chn_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2c')

        deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=32, channel_dim=self.chn_dim,
                                    training=self.phase, name='deconv3a')
        concat3 = tf.concat([deconv3a, conv1], axis=self.chn_dim, name='concat3')
        deconv3b = conv3d_bn_relu(input=concat3, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv3b')
        deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv3c')

        with tf.device("/gpu:1"):
            # auxiliary prediction 0
            if self.up_feat:
                aux0_deconv1 = deconv3d(input=deconv1c, output_chn=32, data_format=self.data_format,
                                        name='aux0_deconv1')
                aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')

                aux0_deconv2 = deconv3d(input=aux0_deconv1_relu, output_chn=32, data_format=self.data_format,
                                        name='aux0_deconv2')
                aux0_deconv2_relu = tf.nn.relu(aux0_deconv2, name='aux0_deconv2_relu')

                aux0_logits = conv3d(input=aux0_deconv2_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_prob')
            else:
                aux0_logits = conv3d(input=deconv1c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_conv')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux0_prob_')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn,
                                       data_format=self.data_format, use_bias=False, use_bilinear=True,
                                       name='aux0_prob')

            # auxiliary prediction 1
            if self.up_feat:
                aux1_deconv1 = deconv3d(input=deconv2c, output_chn=32, data_format=self.data_format,
                                        name='aux1_deconv1')
                aux1_deconv1_relu = tf.nn.relu(aux1_deconv1, name='aux1_deconv1_relu')

                aux1_logits = conv3d(input=aux1_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_prob')
            else:
                aux1_logits = conv3d(input=deconv2c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_conv')
                aux1_logits = deconv3d(input=aux1_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux1_prob')
        # predicted probability
        main_logits = conv3d(input=deconv3c, output_chn=self.output_chn, kernel_size=1, stride=1,
                             data_format=self.data_format, use_bias=True, name='pred_prob')

        return [main_logits, aux1_logits, aux0_logits]

    def UNET_modality32_pool5(self,fusion_method='concat1'):
        #downsample path
        conv1a = [];pool1 = [];conv2a=[];pool2=[];conv3a=[];conv3b=[];pool3=[];conv4a=[];conv4b=[]
        pool4 = [];conv5a = []; conv5b=[]
        for m in range(4):
            with tf.variable_scope('M{}'.format(m)):
                conv1a.append(conv3d_bn_relu(self.input_I[:,:,:,:,m:m+1], 32, 3, 1, self.chn_dim, training=self.phase, name='conv1a'))
                pool1.append(tf.layers.max_pooling3d(conv1a[m],2,2,data_format=self.data_format,name='pool1'))

                with tf.device("/gpu:1"):
                    conv2a.append(conv3d_bn_relu(pool1[m], 64, 3, 1, self.chn_dim, training=self.phase, name='conv2a'))
                    pool2.append(tf.layers.max_pooling3d(conv2a[m],2,2,data_format=self.data_format,name='pool2'))

                    conv3a.append(conv3d_bn_relu(pool2[m], 128, 3, 1, self.chn_dim, training=self.phase, name='conv3a'))
                    conv3b.append(conv3d_bn_relu(conv3a[m], 128, 3, 1, self.chn_dim, training=self.phase, name='conv3b'))
                    pool3.append(tf.layers.max_pooling3d(conv3b[m], 2, 2, data_format=self.data_format, name='pool3'))

                    conv4a.append(conv3d_bn_relu(pool3[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv4a'))
                    conv4b.append(conv3d_bn_relu(conv4a[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv4b'))
                    pool4.append(tf.layers.max_pooling3d(conv4b[m], 2, 2, data_format=self.data_format, name='pool4'))

                    conv5a.append(conv3d_bn_relu(pool4[m], 512, 3, 1, self.chn_dim, training=self.phase, name='conv5a'))
                    conv5b.append(conv3d_bn_relu(conv5a[m], 512, 3, 1, self.chn_dim, training=self.phase, name='conv5b'))

        #upsample path
        with tf.device("/gpu:1"):
            conv5 = self.multimodal_fusion(conv5b, 32, fusion_method, name='conv5_fusion')
            conv4 = self.multimodal_fusion(conv4b,32, fusion_method,name='conv4_fusion')
            conv3 = self.multimodal_fusion(conv3b,32, fusion_method,name='conv3_fusion')
            conv2 = self.multimodal_fusion(conv2a,32, fusion_method,name='conv2_fusion')
        conv1 = self.multimodal_fusion(conv1a,32,  fusion_method,name='conv1_fusion')
        with tf.device("/gpu:1"):
            deconv1a = deconv3d_bn_relu(input=conv5, output_chn=32, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv4], axis=self.chn_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase,  name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=32, channel_dim=self.chn_dim,
                                        training=self.phase,  name='deconv2a')
            concat2 = tf.concat([deconv2a, conv3], axis=self.chn_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2c')

            deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=32, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv3a')
            concat3 = tf.concat([deconv3a, conv2], axis=self.chn_dim, name='concat3')
            deconv3b = conv3d_bn_relu(input=concat3, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase,name='deconv3b')
            deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv3c')

        deconv4a = deconv3d_bn_relu(input=deconv3c, output_chn=32, channel_dim=self.chn_dim,
                                    training=self.phase, name='deconv4a')
        concat4 = tf.concat([deconv4a, conv1], axis=self.chn_dim, name='concat4')
        deconv4b = conv3d_bn_relu(input=concat4, output_chn=32, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv4b')
        deconv4c = conv3d_bn_relu(input=deconv4b, output_chn=32, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv4c')

        with tf.device("/gpu:1"):
            # auxiliary prediction 0
            if self.up_feat:
                aux0_deconv1 = deconv3d(input=deconv1c, output_chn=32, data_format=self.data_format, name='aux0_deconv1')
                aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')

                aux0_deconv2 = deconv3d(input=aux0_deconv1_relu, output_chn=32, data_format=self.data_format,name='aux0_deconv2')
                aux0_deconv2_relu = tf.nn.relu(aux0_deconv2, name='aux0_deconv2_relu')

                aux0_deconv3 = deconv3d(input=aux0_deconv2_relu, output_chn=32, data_format=self.data_format,
                                        name='aux0_deconv3')
                aux0_deconv3_relu = tf.nn.relu(aux0_deconv3, name='aux0_deconv3_relu')

                aux0_logits = conv3d(input=aux0_deconv3_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_prob')
            else:
                aux0_logits = conv3d(input=deconv1c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_conv')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux0_prob_')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn,
                                       data_format=self.data_format, use_bias=False, use_bilinear=True, name='aux0_prob')

            # auxiliary prediction 1
            if self.up_feat:
                aux1_deconv1 = deconv3d(input=deconv2c, output_chn=32, data_format=self.data_format,
                                        name='aux1_deconv1')
                aux1_deconv1_relu = tf.nn.relu(aux1_deconv1, name='aux1_deconv1_relu')

                aux1_deconv2 = deconv3d(input=aux1_deconv1_relu, output_chn=32, data_format=self.data_format,
                                        name='aux1_deconv2')
                aux1_deconv2_relu = tf.nn.relu(aux1_deconv2, name='aux1_deconv2_relu')

                aux1_logits = conv3d(input=aux1_deconv2_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_prob')
            else:
                aux1_logits = conv3d(input=deconv2c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_conv')
                aux1_logits = deconv3d(input=aux1_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux1_prob')
            # auxiliary prediction 2
            if self.up_feat:
                aux2_deconv1 = deconv3d(input=deconv3c, output_chn=32, data_format=self.data_format,
                                        name='aux2_deconv1')
                aux2_deconv1_relu = tf.nn.relu(aux2_deconv1, name='aux2_deconv2_relu')

                aux2_logits = conv3d(input=aux2_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                    data_format=self.data_format, use_bias=True, name='aux2_prob')
        # predicted probability
        main_logits = conv3d(input=deconv4c, output_chn=self.output_chn, kernel_size=1, stride=1,
                             data_format=self.data_format, use_bias=True, name='pred_prob')

        # fusion probability
        fused = tf.concat([deconv4c,aux2_deconv1_relu,aux1_deconv2_relu],axis=self.chn_dim)
        # fused_logits = conv3d(input=fused,output_chn=self.output_chn,kernel_size=1,stride=1,
        #                        data_format=self.data_format,use_bias=True,name='fused_prob')
        return [main_logits, aux2_logits, aux1_logits, aux0_logits]

    def UNET_modality_pool5(self,chn_num = 64, fusion_method='concat1'):
        #downsample path
        conv1a = [];pool1 = [];conv2a=[];pool2=[];conv3a=[];conv3b=[];pool3=[];conv4a=[];conv4b=[]
        pool4 = [];conv5a = []; conv5b=[]
        for m in range(4):
            with tf.variable_scope('M{}'.format(m)):
                with tf.device("/gpu:0"):
                    conv1a.append(conv3d_bn_relu(self.input_I[:,:,:,:,m:m+1], chn_num, 3, 1, self.chn_dim, training=self.phase, name='conv1a'))
                    pool1.append(tf.layers.max_pooling3d(conv1a[m],2,2,data_format=self.data_format,name='pool1'))

                with tf.device("/gpu:1"):
                    conv2a.append(conv3d_bn_relu(pool1[m], chn_num*2, 3, 1, self.chn_dim, training=self.phase, name='conv2a'))
                    pool2.append(tf.layers.max_pooling3d(conv2a[m],2,2,data_format=self.data_format,name='pool2'))

                    conv3a.append(conv3d_bn_relu(pool2[m], chn_num*4, 3, 1, self.chn_dim, training=self.phase, name='conv3a'))
                    conv3b.append(conv3d_bn_relu(conv3a[m],chn_num*4, 3, 1, self.chn_dim, training=self.phase, name='conv3b'))
                    pool3.append(tf.layers.max_pooling3d(conv3b[m], 2, 2, data_format=self.data_format, name='pool3'))

                    conv4a.append(conv3d_bn_relu(pool3[m], chn_num*8, 3, 1, self.chn_dim, training=self.phase, name='conv4a'))
                    conv4b.append(conv3d_bn_relu(conv4a[m],chn_num*8, 3, 1, self.chn_dim, training=self.phase, name='conv4b'))
                    pool4.append(tf.layers.max_pooling3d(conv4b[m], 2, 2, data_format=self.data_format, name='pool4'))

                    conv5a.append(conv3d_bn_relu(pool4[m], chn_num*8, 3, 1, self.chn_dim, training=self.phase, name='conv5a'))
                    conv5b.append(conv3d_bn_relu(conv5a[m],chn_num*8, 3, 1, self.chn_dim, training=self.phase, name='conv5b'))

        #upsample path
        with tf.device("/gpu:1"):
            conv5 = self.multimodal_fusion(conv5b, chn_num, fusion_method, name='conv5_fusion')
            conv4 = self.multimodal_fusion(conv4b, chn_num, fusion_method,name='conv4_fusion')
            conv3 = self.multimodal_fusion(conv3b, chn_num, fusion_method,name='conv3_fusion')
            conv2 = self.multimodal_fusion(conv2a, chn_num, fusion_method,name='conv2_fusion')
        with tf.device("/gpu:0"):
            conv1 = self.multimodal_fusion(conv1a,     chn_num, fusion_method,name='conv1_fusion')

        with tf.device("/gpu:1"):
            deconv1a = deconv3d_bn_relu(input=conv5, output_chn=chn_num, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv4], axis=self.chn_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1, output_chn=chn_num, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=chn_num, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase,  name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=chn_num, channel_dim=self.chn_dim,
                                        training=self.phase,  name='deconv2a')
            concat2 = tf.concat([deconv2a, conv3], axis=self.chn_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=chn_num, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=chn_num, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2c')

            deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=chn_num, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv3a')
            concat3 = tf.concat([deconv3a, conv2], axis=self.chn_dim, name='concat3')
            deconv3b = conv3d_bn_relu(input=concat3, output_chn=chn_num, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase,name='deconv3b')
            deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=chn_num, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv3c')
        with tf.device("/gpu:1"): ### to gpu1 when chn=64
            deconv4a = deconv3d_bn_relu(input=deconv3c, output_chn=chn_num, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv4a')
            concat4 = tf.concat([deconv4a, conv1], axis=self.chn_dim, name='concat4')
            deconv4b = conv3d_bn_relu(input=concat4, output_chn=chn_num, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv4b')
            deconv4c = conv3d_bn_relu(input=deconv4b, output_chn=chn_num, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv4c')

        with tf.device("/gpu:1"):
            # auxiliary prediction 0
            aux0_deconv1 = deconv3d(input=deconv1c, output_chn=chn_num, data_format=self.data_format, name='aux0_deconv1')
            aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')
            aux0_deconv2 = deconv3d(input=aux0_deconv1_relu, output_chn=chn_num, data_format=self.data_format,name='aux0_deconv2')
            aux0_deconv2_relu = tf.nn.relu(aux0_deconv2, name='aux0_deconv2_relu')
            aux0_deconv3 = deconv3d(input=aux0_deconv2_relu, output_chn=chn_num, data_format=self.data_format,name='aux0_deconv3')
            aux0_deconv3_relu = tf.nn.relu(aux0_deconv3, name='aux0_deconv3_relu')
            aux0_logits = conv3d(input=aux0_deconv3_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='aux0_prob')


            # auxiliary prediction 1
            aux1_deconv1 = deconv3d(input=deconv2c, output_chn=chn_num, data_format=self.data_format,name='aux1_deconv1')
            aux1_deconv1_relu = tf.nn.relu(aux1_deconv1, name='aux1_deconv1_relu')
            aux1_deconv2 = deconv3d(input=aux1_deconv1_relu, output_chn=chn_num, data_format=self.data_format,name='aux1_deconv2')
            aux1_deconv2_relu = tf.nn.relu(aux1_deconv2, name='aux1_deconv2_relu')
            aux1_logits = conv3d(input=aux1_deconv2_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='aux1_prob')

            # auxiliary prediction 2
            aux2_deconv1 = deconv3d(input=deconv3c, output_chn=chn_num, data_format=self.data_format,name='aux2_deconv1')
            aux2_deconv1_relu = tf.nn.relu(aux2_deconv1, name='aux2_deconv2_relu')
            aux2_logits = conv3d(input=aux2_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                data_format=self.data_format, use_bias=True, name='aux2_prob')
        with tf.device("/gpu:1"): ### to gpu1 when chn=64
            # predicted probability
            main_logits = conv3d(input=deconv4c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='pred_prob')

        return [main_logits, aux2_logits, aux1_logits, aux0_logits]

    def UNET_modality(self,fusion_method='concat'):
        #downsample path
        conv1a = [];pool1 = [];conv2a=[];pool2=[];conv3a=[];conv3b=[];pool3=[];conv4a=[];conv4b=[]
        for m in range(4):
            with tf.variable_scope('M{}'.format(m)):
                conv1a.append(conv3d_bn_relu(self.input_I[:,:,:,:,m:m+1], 64, 3, 1, self.chn_dim, training=self.phase, name='conv1a'))
                pool1.append(tf.layers.max_pooling3d(conv1a[m],2,2,data_format=self.data_format,name='pool1'))

                with tf.device("/gpu:1"):
                    conv2a.append(conv3d_bn_relu(pool1[m], 128, 3, 1, self.chn_dim, training=self.phase, name='conv2a'))
                    pool2.append(tf.layers.max_pooling3d(conv2a[m],2,2,data_format=self.data_format,name='pool2'))

                    conv3a.append(conv3d_bn_relu(pool2[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv3a'))
                    conv3b.append(conv3d_bn_relu(conv3a[m], 256, 3, 1, self.chn_dim, training=self.phase, name='conv3b'))
                    pool3.append(tf.layers.max_pooling3d(conv3b[m], 2, 2, data_format=self.data_format, name='pool3'))

                    conv4a.append(conv3d_bn_relu(pool3[m], 512, 3, 1, self.chn_dim, training=self.phase, name='conv4a'))
                    conv4b.append(conv3d_bn_relu(conv4a[m], 512, 3, 1, self.chn_dim, training=self.phase, name='conv4b'))

        #upsample path
        with tf.device("/gpu:1"):
            conv4,self.atten4 = self.multimodal_fusion(conv4b,512, fusion_method,name='conv4_fusion')
            conv3,self.atten3 = self.multimodal_fusion(conv3b,256, fusion_method,name='conv3_fusion')
            conv2,self.atten2 = self.multimodal_fusion(conv2a,128, fusion_method,name='conv2_fusion')
        conv1,self.atten1 = self.multimodal_fusion(conv1a,64,  fusion_method,name='conv1_fusion')
        with tf.device("/gpu:1"):
            deconv1a = deconv3d_bn_relu(input=conv4, output_chn=256, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv3], axis=self.chn_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1, output_chn=256, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=256, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=128, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv2a')
            concat2 = tf.concat([deconv2a, conv2], axis=self.chn_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=128, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=128, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2c')

        deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=64, channel_dim=self.chn_dim,
                                    training=self.phase, name='deconv3a')
        concat3 = tf.concat([deconv3a, conv1], axis=self.chn_dim, name='concat3')
        deconv3b = conv3d_bn_relu(input=concat3, output_chn=64, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv3b')
        deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=64, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                  use_bias=False, training=self.phase, name='deconv3c')

        with tf.device("/gpu:1"):
            # auxiliary prediction 0
            if self.up_feat:
                aux0_deconv1 = deconv3d(input=deconv1c, output_chn=128, data_format=self.data_format, name='aux0_deconv1')
                aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')

                aux0_deconv2 = deconv3d(input=aux0_deconv1_relu, output_chn=64, data_format=self.data_format,name='aux0_deconv2')
                aux0_deconv2_relu = tf.nn.relu(aux0_deconv2, name='aux0_deconv2_relu')

                aux0_logits = conv3d(input=aux0_deconv2_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_prob')
            else:
                aux0_logits = conv3d(input=deconv1c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_conv')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux0_prob_')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn,
                                       data_format=self.data_format, use_bias=False, use_bilinear=True, name='aux0_prob')

            # auxiliary prediction 1
            if self.up_feat:
                aux1_deconv1 = deconv3d(input=deconv2c, output_chn=64, data_format=self.data_format,name='aux1_deconv1')
                aux1_deconv1_relu = tf.nn.relu(aux1_deconv1, name='aux1_deconv1_relu')

                aux1_logits = conv3d(input=aux1_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_prob')
            else:
                aux1_logits = conv3d(input=deconv2c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_conv')
                aux1_logits = deconv3d(input=aux1_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux1_prob')
        # predicted probability
        main_logits = conv3d(input=deconv3c, output_chn=self.output_chn, kernel_size=1, stride=1,
                             data_format=self.data_format, use_bias=True, name='pred_prob')

        return [main_logits, aux1_logits, aux0_logits]

    def UNET(self):
        # downsample path
        with tf.device("/gpu:0"):
            with tf.variable_scope('conv1a'):
                conv1a = conv3d(input=self.input_I, output_chn=64, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv1a_bn = tf.layers.batch_normalization(conv1a, axis=self.chn_dim, training=self.phase, name='bn')
                conv1a_relu = tf.nn.relu(conv1a_bn, name='relu')
            pool1 = tf.layers.max_pooling3d(inputs=conv1a_relu, pool_size=2, strides=2, data_format=self.data_format, name='pool1')

            with tf.variable_scope('conv2a'):
                conv2a = conv3d(input=pool1, output_chn=128, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv2a_bn = tf.layers.batch_normalization(conv2a, axis=self.chn_dim, training=self.phase,name='bn')
                conv2a_relu = tf.nn.relu(conv2a_bn, name='relu')
            pool2 = tf.layers.max_pooling3d(inputs=conv2a_relu, pool_size=2, strides=2, data_format=self.data_format, name='pool2')

            with tf.variable_scope('conv3a'):
                conv3a = conv3d(input=pool2, output_chn=256, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv3a_bn = tf.layers.batch_normalization(conv3a, axis=self.chn_dim, training=self.phase,name='bn')
                conv3a_relu = tf.nn.relu(conv3a_bn, name='relu')
            with tf.variable_scope('conv3b'):
                conv3b = conv3d(input=conv3a_relu, output_chn=256, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv3b_bn = tf.layers.batch_normalization(conv3b, axis=self.chn_dim, training=self.phase, name='bn')
                conv3b_relu = tf.nn.relu(conv3b_bn, name='relu')
            pool3 = tf.layers.max_pooling3d(inputs=conv3b_relu, pool_size=2, strides=2, data_format=self.data_format,name='pool3')

            with tf.variable_scope('conv4a'):
                conv4a = conv3d(input=pool3, output_chn=512, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv4a_bn = tf.layers.batch_normalization(conv4a, axis=self.chn_dim, training=self.phase,name='bn')
                conv4a_relu = tf.nn.relu(conv4a_bn, name='relu')
            with tf.variable_scope('conv4b'):
                conv4b = conv3d(input=conv4a_relu, output_chn=512, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False, name='conv')
                conv4b_bn = tf.layers.batch_normalization(conv4b, axis=self.chn_dim, training=self.phase,name='bn')
                conv4b_relu = tf.nn.relu(conv4b_bn, name='relu')

            deconv1a = deconv3d_bn_relu(input=conv4b_relu, output_chn=256, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv3b_relu], axis=self.chn_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1, output_chn=256, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=256, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=128, channel_dim=self.chn_dim,
                                        training=self.phase, name='deconv2a')
            concat2 = tf.concat([deconv2a, conv2a_relu], axis=self.chn_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=128, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=128, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv2c')

            deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=64, channel_dim=self.chn_dim,training=self.phase, name='deconv3a')
            concat3 = tf.concat([deconv3a, conv1a_relu], axis=self.chn_dim, name='concat3')
            deconv3b = conv3d_bn_relu(input=concat3, output_chn=64, kernel_size=1, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv3b')
            deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=64, kernel_size=3, stride=1, channel_dim=self.chn_dim,
                                      use_bias=False, training=self.phase, name='deconv3c')

            # predicted probability
            main_logits = conv3d(input=deconv3c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='pred_prob')

            # auxiliary prediction 0
            if self.up_feat:
                aux0_deconv1 = deconv3d(input=deconv1c, output_chn=128, data_format=self.data_format, name='aux0_deconv1')
                aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')

                aux0_deconv2 = deconv3d(input=aux0_deconv1_relu, output_chn=64, data_format=self.data_format, name='aux0_deconv2')
                aux0_deconv2_relu = tf.nn.relu(aux0_deconv2, name='aux0_deconv2_relu')

                aux0_logits = conv3d(input=aux0_deconv2_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_prob')
            else:
                aux0_logits = conv3d(input=deconv1c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux0_conv')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux0_prob_')
                aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn,
                                       data_format=self.data_format, use_bias=False, use_bilinear=True, name='aux0_prob')

            # auxiliary prediction 1
            if self.up_feat:
                aux1_deconv1 = deconv3d(input=deconv2c, output_chn=64, data_format=self.data_format, name='aux1_deconv1')
                aux1_deconv1_relu = tf.nn.relu(aux1_deconv1, name='aux1_deconv1_relu')

                aux1_logits = conv3d(input=aux1_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_prob')
            else:
                aux1_logits = conv3d(input=deconv2c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                     data_format=self.data_format, use_bias=True, name='aux1_conv')
                aux1_logits = deconv3d(input=aux1_logits, output_chn=self.output_chn, data_format=self.data_format,
                                       use_bias=False, use_bilinear=True, name='aux1_prob')

        return [main_logits, aux1_logits, aux0_logits]


    def DenseNet(self, num_filter = 16,growth_rate = 12,num_layers = [12,12],dropout_rate = 0.2, compression=1.0):
        #Initial Conv 1 (conv0)
        conv1 = conv3d(input=self.input_I, output_chn=num_filter, kernel_size=3, stride=2, data_format=self.data_format,
                       use_bias=False, name='conv0')

        # denseblock 1 (conv1)
        dense1, num_filter = dense_block(conv1, 1, num_layers[0], num_filter, growth_rate, self.chn_dim, dropout_rate,
                                         self.phase)
        tran1, num_filter = transition_block(dense1, 2, num_filter, compression, self.chn_dim, dropout_rate,
                                             self.phase)
        tran1_drop = tf.layers.dropout(tran1, dropout_rate, training=self.phase)
        pool1 = tf.layers.max_pooling3d(tran1_drop, 2, 2, data_format=self.data_format, name="pool1")

        # denseblock 2 (conv2)
        dense2, num_filter = dense_block(pool1, 2, num_layers[1], num_filter, growth_rate, self.chn_dim, dropout_rate,
                                         self.phase)
        tran2, num_filter = transition_block(dense2, 3, num_filter, compression, self.chn_dim, None,
                                             self.phase)

        #upsample
        deconv1 = deconv3d(input=tran2, output_chn=128, data_format=self.data_format, name='deconv1')
        deconv1_relu = tf.nn.relu(deconv1,name='deconv1_relu')
        deconv2 = deconv3d(input=deconv1_relu, output_chn=64, data_format=self.data_format, name='deconv2')
        deconv1_relu = tf.nn.relu(deconv2, name='deconv2_relu')

        # predicted probability
        main_logits = conv3d(input=deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                             data_format=self.data_format, use_bias=True, name='pred_logits')

        # auxiliary prediction 0
        if self.up_feat:
            aux0_deconv1 = deconv3d(input=tran1, output_chn=64, data_format=self.data_format, name='aux0_deconv1')
            aux0_deconv1_relu = tf.nn.relu(aux0_deconv1, name='aux0_deconv1_relu')
            aux0_logits = conv3d(input=aux0_deconv1_relu, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='aux0_logits')
        else:
            aux0_logits = conv3d(input=tran1, output_chn=self.output_chn, kernel_size=1, stride=1,
                                 data_format=self.data_format, use_bias=True, name='aux0_logits_')
            aux0_logits = deconv3d(input=aux0_logits, output_chn=self.output_chn,
                                   data_format=self.data_format, use_bias=False, use_bilinear=True, name='aux0_logits')

        return [main_logits, aux0_logits]

    # train function
    def train(self):
        """Train 3D U-net"""
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load C3D model
        # self.initialize_finetune()

        restore_step = self.load_chkpoint(self.chkpoint_dir)
        # temporary file to save loss
        if restore_step>0:
            loss_log = open(os.path.join(self.model_path, 'loss.txt'), "a")
        else:
            loss_log = open(os.path.join(self.model_path, 'loss.txt'), "w")

        train_imgs, train_labels = load_data_pairs(self.traindata_dir, self.inputI_size, self.reset_h5,"train",all_data=True)
        val_imgs, val_labels = load_data_pairs(self.traindata_dir, self.inputI_size, self.reset_h5, "val")
        print "Total {} volumes for training".format(len(train_imgs))
        print "Total {} volumes for validation".format(len(val_imgs))

        for iter in tqdm(range(self.iter_nums),ncols=50):
            if iter < restore_step:
                continue
            # train batch
            start = time.time()
            batch_img, batch_label = get_batch_patches(train_imgs, train_labels, self.inputI_size, self.batch_size, 4,
                                                       self.data_format, flip_flag=True, rot_flag=True)
            end = time.time()
            #print "time:{}".format(end-start)

            self.sess.run(self.train_op, feed_dict={self.input_I: batch_img, self.input_gt: batch_label})

            if iter%10==0:
                loss,summary_str = self.sess.run([self.entroy_loss,self.summary_op],
                                               feed_dict={self.input_I: batch_img, self.input_gt: batch_label})
                #run validation batch
                val_img, val_label = get_batch_patches(val_imgs, val_labels, self.inputI_size, self.batch_size, 4,
                                                       self.data_format,flip_flag=False, rot_flag=False)
                val_loss = self.sess.run(self.entroy_loss,feed_dict={self.input_I: val_img, self.input_gt: val_label})
                lr = self.sess.run(self.learning_rate)
                loss_log.write("lr: %s train_loss: %s valid_loss:%s\n" % (lr,loss, val_loss))
                print("lr: %.8f train_loss: %.8f valid_loss: %.8f" % (lr,loss, val_loss))

                #add validation loss summary
                val_summary = tf.Summary()
                val_summary.value.add(simple_value=val_loss,tag='losses/val_entroy_loss')
                self.summary_writer.add_summary(val_summary, iter)
                self.summary_writer.add_summary(summary_str, iter)

            if iter%3000 ==0 or iter==self.iter_nums-1:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, iter)

        loss_log.close()


    # test the model
    def test(self, method='vote',dataset='validation'):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load_chkpoint(self.chkpoint_dir)

        # read testing dataset
        if not os.path.exists(self.labeling_dir):
            os.makedirs(self.labeling_dir)

        if dataset == 'validation':
            pair_list = glob('{}/*/*/*_t1.nii.gz'.format(self.testdata_dir))
            pair_list.sort()

            division = pickle.load(open(os.path.join(self.testdata_dir, 'division.pkl'), 'rb'))
            train_list, val_list = division
            list = val_list
        else:
            pair_list = glob('{}/*/*_t1.nii.gz'.format("/home/lqyu/server/gpu8/BRATS2017/data/Brats17ValidationData"))
            pair_list.sort()

            self.labeling_dir = '/home/lqyu/server/gpu8/BRATS2017/validation_result/'+self.labeling_dir.split('/')[-1]
            if not os.path.exists(self.labeling_dir):
                os.makedirs(self.labeling_dir)
            list = None

        for id, T1_path in enumerate(pair_list):
            subject_name = T1_path[:-10]
            if list is not None and subject_name.split('/')[-1] not in list:
                continue

            T1_path = subject_name + "_t1.nii.gz"
            T1c_path = subject_name + "_t1ce.nii.gz"
            T2_path = subject_name + "_t2.nii.gz"
            Flair_path = subject_name + "_flair.nii.gz"

            T1_img = nib.load(T1_path).get_data()
            T1c_img = nib.load(T1c_path).get_data()
            T2_img = nib.load(T2_path).get_data()
            Flair_img = nib.load(Flair_path).get_data().astype('int32')
            img = np.stack((Flair_img, T1_img, T1c_img, T2_img), axis=-1).astype('float32')

            # pre-processing
            # find the bounding box
            pp = np.where(T1_img > 0)
            ss = np.min(pp, axis=1)
            ee = np.max(pp, axis=1)
            img_patch = img[ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2], :]
            for i in range(img.shape[-1]):
                img[:, :, :, i] = (img[:, :, :, i] - np.mean(img_patch[:, :, :, i])) / np.std(img_patch[:, :, :, i])

            subject_name = subject_name.split('/')[-1]
            affine = nib.load(Flair_path).affine

            # decompose volume into list of cubes
            cube_list,idx = decompose_vol2cube(img, self.inputI_size, self.ovlp_ita)

            start_time = time.time()
            patch_results=[]
            attention_results = []
            # cube_list = np.asarray(cube_list)
            # cube_labels = np.zeros((0, self.inputI_size, self.inputI_size, self.inputI_size)).astype('int16')
            # for i in range(0,cube_list.shape[0],8):
            #     ii = min(cube_list.shape[0],i+8)
            #     cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube_list[i:ii]})
            #     for tt in cube_label:
            #         cube_labels.append(tt)
            #     #cube_labels = np.concatenate((cube_labels,cube_label))
            if method=='vote':
                for patch in cube_list:
                    patch_result = self.sess.run([self.pred_label], feed_dict={self.input_I: np.expand_dims(patch,axis=0)})
                    patch_results.append(np.squeeze(patch_result))
                composed_label = compose_label_cube2vol_mvot(patch_results,self.inputI_size,img.shape,4,idx)
            else:
                for patch in cube_list:
                    patch_result = self.sess.run(self.pred_prob, feed_dict={self.input_I: np.expand_dims(patch,axis=0)})
                    patch_results.append(np.squeeze(patch_result))
                composed_label = compose_label_cube2vol_mean(patch_results,self.inputI_size,img.shape,4,idx)
            composed_label = composed_label.astype('int16')
            composed_label[composed_label == 3] = 4

            # remove minor connected components
            post_pred = post_prediction(composed_label)
            end_time = time.time()
            print "{}: {}".format(id, end_time - start_time)

            #save post processed prediction
            if dataset == 'validation':
                labeling_path = os.path.join(self.labeling_dir, subject_name+'_seg.nii.gz')
            else:
                labeling_path = os.path.join(self.labeling_dir, subject_name + '.nii.gz')

            labeling_vol = nib.Nifti1Image(post_pred, affine)
            nib.save(labeling_vol, labeling_path)

            #save original prediction
            if not os.path.exists(self.labeling_dir+"_ori"):
                os.makedirs(self.labeling_dir+"_ori")
            if dataset == 'validation':
                labeling_path = os.path.join(self.labeling_dir+"_ori", subject_name + '_seg.nii.gz')
            else:
                labeling_path = os.path.join(self.labeling_dir + "_ori", subject_name + '.nii.gz')
            labeling_vol = nib.Nifti1Image(composed_label, affine)
            nib.save(labeling_vol, labeling_path)

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s" % (self.batch_size, self.inputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):

        model_dir = "%s_%s" % (self.batch_size, self.inputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return step
        else:
            print(" [!] Load checkpoint failed... No checkpoint is found")
            return 0

    # load C3D model
    def initialize_finetune(self):
        checkpoint_dir = '../outcome/model/C3D_unet_1chn'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_ft.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))