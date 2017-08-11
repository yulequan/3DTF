import os
import shutil
import tensorflow as tf

from ini_file_io import load_train_ini
from model import Model

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(_):
    # load training parameter #
    model_path = "../model/UNet_feat_mm_attention"
    param_sets = load_train_ini(os.path.join(model_path,'ini','tr_param.ini'))
    param_set = param_sets[0]
    param_set['model_path'] = model_path
    param_set['chkpoint_dir'] = os.path.join(model_path,'chkpoint')
    param_set['up_feat'] = True
    param_set['reset_h5'] = False
    param_set['data_format'] = 'channels_last'

    param_set['testdata_dir'] = '/home/lqyu/server/gpu8/BRATS2017/data/Brats17TrainingData'
    param_set['labeling_dir'] = '/home/lqyu/server/gpu8/BRATS2017/result/UNet_feat_mm_attention'
    param_set['phase'] = 'train'

    print '====== Parameters >>> %s <<< ======' % param_set['phase']
    print param_set
    print '=============================='

    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = Model(sess, param_set)

        if param_set['phase'] == 'train':
            # copy the model
            if not os.path.exists(os.path.join(model_path, 'code')):
                os.makedirs(os.path.join(model_path, 'code'))
            shutil.copy('model.py', os.path.join(model_path, 'code', 'model.py'))
            shutil.copy('main.py', os.path.join(model_path, 'code', 'main.py'))
            shutil.copy('ops.py', os.path.join(model_path, 'code', 'ops.py'))
            shutil.copy('utils.py', os.path.join(model_path, 'code', 'utils.py'))

            model.train()
        elif param_set['phase'] == 'test':
            model.test()

if __name__ == '__main__':
    tf.app.run()
