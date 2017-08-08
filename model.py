from __future__ import division
import time
from tqdm import tqdm
from ops import *
from utils import *
from seg_eval import *
class unet_3D_xy(object):
    """ Implementation of 3D U-net"""
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
        self.beta1          = param_set['beta1']
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
        self.build_model(up_feat=self.up_feat)

        # trainable variables
        self.u_vars = tf.trainable_variables()

        # extract the layers for fine tuning
        ft_layer = ['conv1a/conv/kernel:0',
                    'conv2a/conv/kernel:0',
                    'conv3a/conv/kernel:0',
                    'conv3b/conv/kernel:0',
                    'conv4a/conv/kernel:0',
                    'conv4b/conv/kernel:0']

        self.ft_vars = []
        for var in self.u_vars:
            for k in range(len(ft_layer)):
                if ft_layer[k] in var.name:
                    self.ft_vars.append(var)
                    break

        # create model saver
        self.saver = tf.train.Saver()
        # saver to load pre-trained C3D model
        self.saver_ft = tf.train.Saver(self.ft_vars)

    # build model graph : 3D-Unet
    def build_model(self,up_feat=True):
        phase_flag = (self.phase=='train')
        if self.data_format=='channels_last':
            channel_dim = 4
            #input
            self.input_I = tf.placeholder(dtype=tf.float32, shape=[None, self.inputI_size, self.inputI_size,
                                                                   self.inputI_size, self.inputI_chn], name='inputI')
        else:
            channel_dim = 1
            # input
            self.input_I = tf.placeholder(dtype=tf.float32, shape=[None, self.inputI_chn, self.inputI_size,
                                                                   self.inputI_size, self.inputI_size], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int32, shape=[None, self.inputI_size, self.inputI_size,
                                                         self.inputI_size], name='target')
        #downsample path
        with tf.device("/gpu:0"):
            with tf.variable_scope('conv1a'):
                conv1a = conv3d(input=self.input_I, output_chn=64, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv1a_bn = tf.layers.batch_normalization(conv1a,axis=channel_dim,training=phase_flag,name='bn')
                conv1a_relu = tf.nn.relu(conv1a_bn,name='relu')
            pool1 = tf.layers.max_pooling3d(inputs=conv1a_relu, pool_size=2, strides=2, data_format=self.data_format, name='pool1')

            with tf.variable_scope('conv2a'):
                conv2a = conv3d(input=pool1, output_chn=128, kernel_size=3, stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv2a_bn = tf.layers.batch_normalization(conv2a,axis=channel_dim,training=phase_flag,name='bn')
                conv2a_relu = tf.nn.relu(conv2a_bn,name='relu')
            pool2 = tf.layers.max_pooling3d(inputs=conv2a_relu, pool_size=2, strides=2, data_format=self.data_format, name='pool2')

            with tf.variable_scope('conv3a'):
                conv3a = conv3d(input=pool2,output_chn=256,kernel_size=3,stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv3a_bn = tf.layers.batch_normalization(conv3a,axis=channel_dim,training=phase_flag,name='bn')
                conv3a_relu = tf.nn.relu(conv3a_bn,name='relu')
            with tf.variable_scope('conv3b'):
                conv3b = conv3d(input=conv3a_relu,output_chn=256,kernel_size=3,stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv3b_bn = tf.layers.batch_normalization(conv3b,axis=channel_dim,training=phase_flag,name='bn')
                conv3b_relu = tf.nn.relu(conv3b_bn,name='relu')
            pool3 = tf.layers.max_pooling3d(inputs=conv3b_relu, pool_size=2, strides=2, data_format=self.data_format, name='pool3')

            with tf.variable_scope('conv4a'):
                conv4a = conv3d(input=pool3,output_chn=512,kernel_size=3,stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv4a_bn = tf.layers.batch_normalization(conv4a,axis=channel_dim,training=phase_flag,name='bn')
                conv4a_relu = tf.nn.relu(conv4a_bn,name='relu')
            with tf.variable_scope('conv4b'):
                conv4b = conv3d(input=conv4a_relu,output_chn=512,kernel_size=3,stride=1, data_format=self.data_format,
                                use_bias=False,name='conv')
                conv4b_bn = tf.layers.batch_normalization(conv4b,axis=channel_dim,training=phase_flag,name='bn')
                conv4b_relu = tf.nn.relu(conv4b_bn,name='relu')

        #upsampling path
        with tf.device("/gpu:0"):
            deconv1a = deconv3d_bn_relu(input=conv4b_relu, output_chn=256, channel_dim=channel_dim,
                                        training=phase_flag, name='deconv1a')
            concat1 = tf.concat([deconv1a, conv3b_relu], axis=channel_dim, name='concat1')
            deconv1b = conv3d_bn_relu(input=concat1,output_chn=256,kernel_size=1,stride=1,channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv1b')
            deconv1c = conv3d_bn_relu(input=deconv1b, output_chn=256, kernel_size=3, stride=1, channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv1c')

            deconv2a = deconv3d_bn_relu(input=deconv1c, output_chn=128, channel_dim=channel_dim,
                                        training=phase_flag, name='deconv2a')
            concat2 = tf.concat([deconv2a, conv2a_relu], axis=channel_dim, name='concat2')
            deconv2b = conv3d_bn_relu(input=concat2, output_chn=128, kernel_size=1, stride=1, channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv2b')
            deconv2c = conv3d_bn_relu(input=deconv2b, output_chn=128, kernel_size=3, stride=1, channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv2c')

            deconv3a = deconv3d_bn_relu(input=deconv2c, output_chn=64, channel_dim=channel_dim,
                                        training=phase_flag, name='deconv3a')
            concat3 = tf.concat([deconv3a, conv1a_relu], axis=channel_dim, name='concat3')
            deconv3b = conv3d_bn_relu(input=concat3, output_chn=64, kernel_size=1, stride=1, channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv3b')
            deconv3c = conv3d_bn_relu(input=deconv3b, output_chn=64, kernel_size=3, stride=1, channel_dim=channel_dim,
                                      use_bias=False, training=phase_flag, name='deconv3c')


            #============================

            # predicted probability
            self.pred_prob = conv3d(input=deconv3c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                    data_format=self.data_format, use_bias=True, name='pred_prob')

            if self.data_format=='channels_first':
                self.pred_prob = tf.transpose(self.pred_prob, perm=(0, 2, 3, 4, 1))

            # predicted labels
            self.soft_prob = tf.nn.softmax(self.pred_prob, name='pred_soft')
            self.pred_label = tf.argmax(self.soft_prob, axis=4, name='argmax')

            if not phase_flag:
                return

            # auxiliary prediction 0
            if up_feat:
                aux0_deconv_a = deconv3d_bn_relu(input=deconv1c, output_chn=64, channel_dim=channel_dim,
                                                 training=phase_flag, name='aux0_deconv_a')
                aux0_deconv_b = deconv3d_bn_relu(input=aux0_deconv_a, output_chn=64, channel_dim=channel_dim,
                                                 training=phase_flag, name='aux0_deconv_b')
                self.aux0_prob = conv3d(input=aux0_deconv_b, output_chn=self.output_chn, kernel_size=1, stride=1,
                                        data_format=self.data_format, use_bias=True, name='aux0_prob')
            else:
                aux0_prob = conv3d(input=deconv1c,output_chn=self.output_chn,kernel_size=1,stride=1, data_format=self.data_format,
                                use_bias=True, name='aux0_conv')
                aux0_prob_ = deconv3d(input=aux0_prob, output_chn=self.output_chn,data_format=self.data_format,
                                                 use_bias=False, use_bilinear=True, name='aux0_prob_')
                self.aux0_prob = deconv3d(input=aux0_prob_, output_chn=self.output_chn, data_format=self.data_format,
                                     use_bias=False, use_bilinear=True, name='aux0_prob')

            # auxiliary prediction 1
            if up_feat:
                aux1_deconv_a = deconv3d_bn_relu(input=deconv2c, output_chn=64, channel_dim=channel_dim,
                                                 training=phase_flag, name='aux1_deconv_a')
                self.aux1_prob = conv3d(input=aux1_deconv_a, output_chn=self.output_chn, kernel_size=1, stride=1,
                                        data_format=self.data_format, use_bias=True, name='aux1_prob')
            else:
                aux1_prob = conv3d(input=deconv2c, output_chn=self.output_chn, kernel_size=1, stride=1,
                                   data_format=self.data_format,
                                   use_bias=True, name='aux1_conv')
                self.aux1_prob = deconv3d(input=aux1_prob, output_chn=self.output_chn, data_format=self.data_format,
                                      use_bias=False, use_bilinear=True, name='aux1_prob')

            # ========= calculate loss
            if self.data_format=='channels_first':
                self.aux0_prob = tf.transpose(self.aux0_prob, perm=(0, 2, 3, 4, 1))
                self.aux1_prob = tf.transpose(self.aux1_prob, perm=(0, 2, 3, 4, 1))
            self.main_dice_loss = tf.losses.sparse_softmax_cross_entropy(self.input_gt,self.pred_prob,weights=0.625)
            self.aux0_dice_loss = tf.losses.sparse_softmax_cross_entropy(self.input_gt, self.aux0_prob,weights=0.125)
            self.aux1_dice_loss = tf.losses.sparse_softmax_cross_entropy(self.input_gt, self.aux1_prob,weights=0.250)

            #apply regularization
            tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.0005),tf.trainable_variables())
            self.total_loss = tf.losses.get_total_loss(name='total_loss')

        t1 = tf.squeeze(tf.slice(self.input_I, [0, 0, 0, 40, 0], [-1, -1, -1, 1, 1]),axis=-1)
        pred = tf.cast(tf.slice(self.pred_label,[0,0,0,40],[-1,-1,-1,1])*80,tf.uint8)
        gt = tf.cast(tf.slice(self.input_gt,[0,0,0,40],[-1,-1,-1,1])*80,tf.uint8)
        tf.summary.image('img/input',t1,max_outputs=3)
        tf.summary.image('img/pred', pred, max_outputs=3)
        tf.summary.image('img/gt', gt, max_outputs=3)
        tf.summary.scalar('losses/regu_loss', tf.losses.get_regularization_loss())
        tf.summary.scalar('losses/main_loss', self.main_dice_loss)
        tf.summary.scalar('losses/aux0_loss', self.aux0_dice_loss)
        tf.summary.scalar('losses/aux1_loss', self.aux1_dice_loss)
        tf.summary.scalar('losses/total_loss', self.total_loss)

    # train function
    def train(self):
        """Train 3D U-net"""
        # aa = tf.placeholder(tf.float32,(1,80,80,80,1))
        # bb = deconv3d(input=aa, output_chn=1,use_bias=False, use_bilinear=True, name='aaa')

        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = tf.contrib.training.create_train_op(self.total_loss,u_optimizer)
        summary_op = tf.summary.merge_all()

        self.summary_writer = tf.summary.FileWriter(os.path.join(self.model_path,'logs'),self.sess.graph)

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

        train_imgs, train_labels = load_data_pairs(self.traindata_dir, self.inputI_size, self.reset_h5,"train")
        val_imgs, val_labels = load_data_pairs(self.traindata_dir, self.inputI_size, self.reset_h5, "val")
        print "Total {} volumes for training".format(len(train_imgs))
        print "Total {} volumes for validation".format(len(val_imgs))

        for iter in tqdm(range(self.iter_nums),ncols=70):
            if iter < restore_step:
                continue
            # train batch
            batch_img, batch_label = get_batch_patches(train_imgs, train_labels, self.inputI_size, self.batch_size, chn=4, flip_flag=True, rot_flag=True)
            if self.data_format=='channels_first':
                batch_img = np.transpose(batch_img,(0,4,1,2,3))
            self.sess.run(train_op, feed_dict={self.input_I: batch_img, self.input_gt: batch_label})

            if iter%10==0:
                loss,summary_str = self.sess.run([self.total_loss,summary_op],
                                               feed_dict={self.input_I: batch_img, self.input_gt: batch_label})
                #run validation batch
                val_img, val_label = get_batch_patches(val_imgs, val_labels, self.inputI_size, self.batch_size, chn=4,
                                                       flip_flag=False, rot_flag=False)
                if self.data_format == 'channels_first':
                    val_img = np.transpose(val_img, (0, 4, 1, 2, 3))
                val_loss = self.sess.run(self.total_loss,feed_dict={self.input_I: val_img, self.input_gt: val_label})

                loss_log.write("train_loss: %s\tvalid_loss:%s\n" % (loss, val_loss))
                print("train_loss: %.8f, valid_loss: %.8f" % (loss, val_loss))
                self.summary_writer.add_summary(summary_str, iter)

            if iter%3000 ==2999 or iter==self.iter_nums-1:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, iter)

        loss_log.close()


    # test the model
    def test(self):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load_chkpoint(self.chkpoint_dir)

        # read testing dataset
        if not os.path.exists(self.labeling_dir):
            os.makedirs(self.labeling_dir)

        pair_list = glob('{}/*/*/*_t1.nii.gz'.format(self.testdata_dir))
        pair_list.sort()
        for id, T1_path in enumerate(pair_list[::5]):
            subject_name = T1_path[:-10]
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
            # predict on each cube
            start_time = time.time()
            patch_results=[]

            # cube_list = np.asarray(cube_list)
            # cube_labels = np.zeros((0, self.inputI_size, self.inputI_size, self.inputI_size)).astype('int16')
            # for i in range(0,cube_list.shape[0],8):
            #     ii = min(cube_list.shape[0],i+8)
            #     cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube_list[i:ii]})
            #     for tt in cube_label:
            #         cube_labels.append(tt)
            #     #cube_labels = np.concatenate((cube_labels,cube_label))
            for patch in cube_list:
                patch_result = self.sess.run(self.soft_prob, feed_dict={self.input_I: np.expand_dims(patch,axis=0)})
                patch_results.append(np.squeeze(patch_result))

            # compose cubes into a volume
            composed_label = compose_label_cube2vol_mean(patch_results,self.inputI_size,img.shape,4,idx)
            composed_label = composed_label.astype('int16')
            composed_label[composed_label == 3] = 4

            # remove minor connected components
            post_pred = post_prediction(composed_label)

            end_time = time.time()
            print "{}: {}".format(id, end_time - start_time)

            #save post processed prediction
            labeling_path = os.path.join(self.labeling_dir, subject_name+'_seg.nii.gz')
            labeling_vol = nib.Nifti1Image(post_pred, affine)
            nib.save(labeling_vol, labeling_path)

            #save original prediction
            if not os.path.exists(self.labeling_dir+"_ori"):
                os.makedirs(self.labeling_dir+"_ori")
            labeling_path = os.path.join(self.labeling_dir+"_ori", subject_name + '_seg.nii.gz')
            labeling_vol = nib.Nifti1Image(composed_label, affine)
            nib.save(labeling_vol, labeling_path)

    # test function for cross validation
    def test4crsv(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # all dice
        all_dice = np.zeros([int(len(test_list)/2), 8])

        # test
        for k in range(0, len(test_list), 2):
            # load the volume
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita, self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')
            print np.unique(composed_label)

            # for s in range(composed_label.shape[2]):
            #     cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]*255.0).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
            #     cv2.waitKey(30)

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            labeling_path = os.path.join(self.labeling_dir, ('test_' + str(k) + '.nii.gz'))
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

            # evaluation
            gt_file = nib.load(test_list[k + 1])
            gt_label = gt_file.get_data().copy()
            k_dice_c = seg_eval_metric(composed_label_resz, gt_label)
            print k_dice_c
            all_dice[int(k/2), :] = np.asarray(k_dice_c)

        mean_dice = np.mean(all_dice, axis=0)
        print "average dice: "
        print mean_dice

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