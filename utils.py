from __future__ import division
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate
import h5py
import os
from scipy.ndimage import measurements
from glob import glob
import pickle
import tensorflow as tf


##############################
# process .nii data
# load data according to the data path
# we need to do some preprocessing step if necessary

def load_test_data(testingdata_dir,reset_h5 = False):
    pair_list = glob('{}/*/*_t1.nii.gz'.format(testingdata_dir))
    pair_list.sort()

    img_clec = []
    subject_names = []
    affines = []

    for id, T1_path in enumerate(pair_list[:5]):
        print id
        subject_name = T1_path[:-10]
        T1_path = subject_name+"_t1.nii.gz"
        T1c_path = subject_name+"_t1ce.nii.gz"
        T2_path = subject_name+"_t2.nii.gz"
        Flair_path = subject_name+"_flair.nii.gz"

        T1_img = nib.load(T1_path).get_data()
        T1c_img = nib.load(T1c_path).get_data()
        T2_img = nib.load(T2_path).get_data()
        Flair_img = nib.load(Flair_path).get_data().astype('int32')
        img = np.stack((Flair_img,T1_img,T1c_img,T2_img),axis=-1).astype('float32')

        #pre-processing
        # find the bounding box
        pp = np.where(T1_img > 0)
        ss = np.min(pp, axis=1)
        ee = np.max(pp, axis=1)
        img_patch = img[ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2], :]
        for i in range(img.shape[-1]):
            img[:, :, :, i] = (img[:, :, :, i] - np.mean(img_patch[:, :, :, i])) / np.std(img_patch[:, :, :, i])

        img_clec.append(img)
        subject_names.append(subject_name.split('/')[-1])
        affines.append(nib.load(Flair_path).affine)
    return img_clec,subject_names,affines


def load_data_pairs(traindata_dir, patch_dim=64, reset_h5=False, phase='train',all_data = True):
    """load all volume pairs"""
    pair_list = glob('{}/*/*/*_seg.nii.gz'.format(traindata_dir))
    pair_list.sort()

    division = pickle.load(open(os.path.join(traindata_dir,'division.pkl'),'rb'))
    train_list, val_list = division
    list = train_list if phase=='train' else val_list

    img_clec = []
    label_clec = []

    file_disk = os.path.join(traindata_dir, phase+'_data.h5')
    if os.path.exists(file_disk) and not reset_h5:
        disk_buffer = h5py.File(file_disk,'r')
        h5_size = disk_buffer['size']
        imgs = disk_buffer['img']
        labels = disk_buffer['gt']

        for id in range(h5_size.shape[0]):
            img_clec.append(np.reshape(imgs[id],tuple(h5_size[id])+(4,)))
            label_clec.append(np.reshape(labels[id],tuple(h5_size[id])))

        # add the val data if we use all data to train
        if all_data and phase=='train':
            print "also use val data to train"
            file_disk = os.path.join(traindata_dir,'val_data.h5')
            disk_buffer = h5py.File(file_disk, 'r')
            h5_size = disk_buffer['size']
            imgs = disk_buffer['img']
            labels = disk_buffer['gt']

            for id in range(h5_size.shape[0]):
                img_clec.append(np.reshape(imgs[id], tuple(h5_size[id]) + (4,)))
                label_clec.append(np.reshape(labels[id], tuple(h5_size[id])))
    else:
        disk_buffer = h5py.File(file_disk,'w')
        dt1 = h5py.special_dtype(vlen=np.dtype('float32'))
        dt2 = h5py.special_dtype(vlen=np.dtype('int32'))
        h5_image = disk_buffer.create_dataset('img',(1000,),dtype=dt1)
        h5_gt = disk_buffer.create_dataset('gt',(1000,),dtype=dt2)
        img_size = []
        img_name = []
        cnt = -1
        for id,seg_path in enumerate(pair_list):
            subject_name = seg_path[:-11]
            if subject_name.split('/')[-1] not in list:
                continue
            print id
            cnt +=1
            T1_path = subject_name+"_t1.nii.gz"
            T1c_path = subject_name+"_t1ce.nii.gz"
            T2_path = subject_name+"_t2.nii.gz"
            Flair_path = subject_name+"_flair.nii.gz"

            T1_img = nib.load(T1_path).get_data()
            T1c_img = nib.load(T1c_path).get_data()
            T2_img = nib.load(T2_path).get_data()
            Flair_img = nib.load(Flair_path).get_data().astype('int32')
            img = np.stack((Flair_img,T1_img,T1c_img,T2_img),axis=-1).astype('float32')
            seg = nib.load(seg_path).get_data()

            # find the bounding box
            pp = np.where(seg>0)
            ss = np.min(pp,axis=1)
            ee = np.max(pp,axis=1)
            agu = np.maximum(np.ceil((patch_dim -(ee-ss))),0)
            sample_size = np.maximum(patch_dim+3,ee-ss)
            ss = np.maximum(ss-agu,0).astype('int32')
            ee = np.minimum(ss+sample_size,seg.shape).astype('int32')

            img = img[ss[0]:ee[0],ss[1]:ee[1],ss[2]:ee[2],:]
            seg = seg[ss[0]:ee[0],ss[1]:ee[1],ss[2]:ee[2]]

            for i in range(img.shape[-1]):
                img[:,:,:,i] = (img[:,:,:,i]-np.mean(img[:,:,:,i]))/np.std(img[:,:,:,i])
            seg[seg==4]=3

            img_clec.append(img)
            label_clec.append(seg)
            h5_image[cnt]= img.flatten()
            h5_gt[cnt]= seg.flatten()
            img_size.append(ee - ss)
            img_name.append(subject_name.split('/')[-1])
        disk_buffer.create_dataset('size',data=img_size)
        disk_buffer.create_dataset('name',data=img_name)

    disk_buffer.close()
    return img_clec, label_clec


def get_batch_patches(img_clec, label_clec, patch_dim, batch_size, chn, data_format = 'channels_last',flip_flag=True, rot_flag=True):

    """generate a batch of paired patches for training"""
    batch_img = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, chn]).astype('float32')
    batch_label = np.zeros([batch_size, patch_dim, patch_dim, patch_dim]).astype('int32')

    for k in range(batch_size):
        # randomly select an image pair
        rand_idx = np.random.randint(len(img_clec))
        rand_img = img_clec[rand_idx]
        rand_label = label_clec[rand_idx]
        rand_img = rand_img.astype('float32')
        rand_label = rand_label.astype('int32')

        # randomly select a box anchor
        l, w, h = rand_label.shape
        assert l>patch_dim
        assert w>patch_dim
        assert h>patch_dim

        ll = np.random.randint(l - patch_dim)
        ww = np.random.randint(w - patch_dim)
        hh = np.random.randint(h - patch_dim)

        # crop
        img_temp = rand_img[ll:ll+patch_dim, ww:ww+patch_dim, hh:hh+patch_dim,:]
        label_temp = rand_label[ll:ll+patch_dim, ww:ww+patch_dim, hh:hh+patch_dim]

        # possible augmentation

        # rotation
        if rot_flag and np.random.random() > 0.5:
            # print 'rotating patch...'
            angle = np.random.randint(4)*90
            img_temp = rotate(img_temp, angle=angle, axes=(1, 0), reshape=False, order=1)
            label_temp = rotate(label_temp, angle=angle, axes=(1, 0), reshape=False, order=0)

        # flip
        if flip_flag and np.random.random()>0.5:
            img_temp = np.flip(img_temp,axis=0)
            label_temp = np.flip(label_temp,axis=0)

        batch_img[k, :, :, :,:] = img_temp
        batch_label[k, :, :, :] = label_temp

    if data_format == 'channels_first':
        batch_img = np.transpose(batch_img, (0, 4, 1, 2, 3))
    return batch_img, batch_label

# decompose volume into list of cubes
def decompose_vol2cube(vol_data, cube_size, ita):
    dim = np.asarray(vol_data.shape)
    fold = np.floor(dim/cube_size)+ita
    overlap = np.ceil((dim-cube_size)/(fold-1)).astype('int32')

    idx1 = range(0,dim[0]-cube_size,overlap[0])+[dim[0]-cube_size]
    idx2 = range(0, dim[1] - cube_size, overlap[1]) + [dim[1] - cube_size]
    idx3 = range(0, dim[2] - cube_size, overlap[2]) + [dim[2] - cube_size]

    cube_list = []
    for id1 in idx1:
        for id2 in idx2:
            for id3 in idx3:
                cube_list.append(vol_data[id1:id1+cube_size, id2:id2+cube_size, id3:id3+cube_size,:].copy())
    return cube_list, (idx1,idx2,idx3)


# compose list of label cubes into a label volume
def compose_label_cube2vol_mvot(cube_labels, cube_size, dim, n_class, idx):

    idx1,idx2,idx3 = idx
    cnt = np.zeros(dim[0:3] + (n_class,))
    counter = 0
    for id1 in idx1:
        for id2 in idx2:
            for id3 in idx3:
                m1,m2,m3 = np.mgrid[id1:id1+cube_size, id2:id2+cube_size, id3:id3+cube_size]
                cnt[m1, m2, m3, cube_labels[counter]] += 1
                counter +=1
                #another implementation
                # for k in range(n_class):
                #     idx_classes_mat[:, :, :, k] = (cube_list[counter] == k)
                # cnt[id1:id1+cube_size, id2:id2+cube_size, id3:id3+cube_size,:] += idx_classes_mat

    label = np.argmax(cnt,axis=-1)
    return label

# compose list of label cubes into a label volume
def compose_label_cube2vol_mean(cube_score, cube_size, dim, n_class, idx):

    idx1,idx2,idx3 = idx
    score = np.zeros(dim[0:3] + (n_class,))
    cnt = np.zeros(dim[0:3])
    counter = 0
    for id1 in idx1:
        for id2 in idx2:
            for id3 in idx3:
                score[id1:id1+cube_size, id2:id2+cube_size, id3:id3+cube_size,:] +=cube_score[counter]
                cnt[id1:id1+cube_size, id2:id2+cube_size, id3:id3+cube_size] +=1
                counter +=1
    score = score/np.tile(np.expand_dims(cnt,axis=-1),(1,1,1,n_class))
    label = np.argmax(score,axis=-1)
    return label


def post_prediction(prediction):
    enhance = (prediction==4)
    core = np.logical_or(prediction==1, prediction==4)
    whole = np.logical_or(np.logical_or(prediction==1, prediction==2),prediction==4)

    whole = remove_minor_cc(whole)

    post_pred = np.zeros_like(prediction)
    post_pred[whole==1] = 2
    post_pred[np.logical_and(whole==1,core==1)]=1
    post_pred[np.logical_and(whole==1, np.logical_and(core==1, enhance==1))] = 4

    return post_pred



def remove_minor_cc(data, rej_ratio=0.2):
    """Remove small connected components refer to rejection ratio"""
    label, num = measurements.label(data,structure=np.ones((3,3,3)))
    total = np.sum(data>0)
    for cc in range(1,num+1):
        single = np.sum(label==cc)
        if single < total*rej_ratio:
            data[label==cc]=0
    return data


