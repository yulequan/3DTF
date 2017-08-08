import os
import tensorflow as tf

from ini_file_io import load_train_ini
from model import unet_3D_xy

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main(_):
    # load training parameter #
    model_path = "../model/Model_feat"
    param_sets = load_train_ini(os.path.join(model_path,'ini','tr_param.ini'))
    param_set = param_sets[0]
    param_set['model_path'] = model_path
    param_set['chkpoint_dir'] = os.path.join(model_path,'chkpoint')
    param_set['reset_h5'] = False
    param_set['data_format'] = 'channels_last' #'channels_first'
    param_set['phase'] ='train'
    param_set['testdata_dir'] = '/home/lqyu/server/gpu8/BRATS2017/data/Brats17TrainingData'
    param_set['labeling_dir'] = '/home/lqyu/server/gpu8/BRATS2017/data/result_s_avg'
    param_set['up_feat'] = True

    print '=======Parameters============='
    print '====== Phase >>> %s <<< ======' % param_set['phase']
    print param_set
    print '=============================='

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])

    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = unet_3D_xy(sess, param_set)

        if param_set['phase'] == 'train':
            model.train()
        elif param_set['phase'] == 'test':
            model.test()
        elif param_set['phase'] == 'crsv':
            model.test4crsv()

if __name__ == '__main__':
    tf.app.run()
