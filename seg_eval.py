import numpy as np
import time
import nibabel as nib
import medpy.metric.binary as metric
import os
from glob import glob
import pickle

def eval_model():
    testdata_dir = '/home/lqyu/server/gpu8/BRATS2017/data/Brats17TrainingData'
    labeling_dir = '/home/lqyu/server/gpu8/BRATS2017/result/Dense_feat_ori'

    pair_list = glob('{}/*/*/*_seg.nii.gz'.format(testdata_dir))
    pair_list.sort()

    division = pickle.load(open(os.path.join(testdata_dir, 'division.pkl'), 'rb'))
    train_list, val_list = division
    list = val_list


    results = []
    for id, seg_path in enumerate(pair_list):
        start = time.time()
        subject_name = seg_path[:-11]
        if subject_name.split('/')[-1] not in list:
            continue

        gt = nib.load(seg_path).get_data()
        pred_path = os.path.join(labeling_dir, seg_path.split('/')[-1])
        pred = nib.load(pred_path).get_data()
        m = eval_result(pred, gt)
        print m
        results.append(m)
        end = time.time()
        #print end-start

    results = np.asarray(results)
    print "=============Mean==========="
    print "Dice_ET\tSen_ET\tDice_WT\tSen_WT\tDice_TC\tSen_TC"
    print np.mean(results,axis=0)

def eval_seg_metric(pred, gt):
    dice    = metric.dc(pred,gt)
    #hd     = metric.hd(pred,gt)
    #assd   = metric.assd(pred,gt)
    sens    = metric.recall(pred,gt)

    return dice, sens

def eval_result(pred,gt):
    pred_enhance = (pred==4)
    gt_enhance = (gt==4)
    ET = eval_seg_metric(pred_enhance, gt_enhance)

    pred_core = (np.logical_or(pred==1, pred==4))
    gt_core = (np.logical_or(gt==1, gt==4))
    TC = eval_seg_metric(pred_core, gt_core)

    pred_whole = np.logical_or(np.logical_or(pred==1, pred==2),pred==4)
    gt_whole = np.logical_or(np.logical_or(gt==1, gt==2),gt==4)
    WT = eval_seg_metric(pred_whole, gt_whole)

    return ET+WT+TC


# calculate evaluation metrics for segmentation
def seg_eval_metric(pred_label, gt_label):
    class_n = np.unique(gt_label)
    # dice
    dice_c = dice_n_class(move_img=pred_label, refer_img=gt_label)
    # # conformity
    # conform_c = conform_n_class(move_img=pred_label, refer_img=gt_label)
    # # jaccard
    # jaccard_c = jaccard_n_class(move_img=pred_label, refer_img=gt_label)
    # # precision and recall
    # precision_c, recall_c = precision_recall_n_class(move_img=pred_label, refer_img=gt_label)

    # return dice_c, conform_c, jaccard_c, precision_c, recall_c
    return dice_c

# dice value
def dice_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    dice_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c


# conformity value
def conform_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    conform_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        # dice
        dice_temp = (2.0 * ints) / sums
        # conformity
        conform_temp = (3*dice_temp - 2) / dice_temp

        conform_c.append(conform_temp)

    return conform_c


# Jaccard index
def jaccard_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    jaccard_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # union
        uni = np.sum(np.logical_or(move_img_c, refer_img_c)*1) + 0.0001

        jaccard_c.append(ints / uni)

    return jaccard_c


# precision and recall
def precision_recall_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    precision_c = []
    recall_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # precision
        prec = ints / (np.sum(move_img_c*1) + 0.001)
        # recall
        recall = ints / (np.sum(refer_img_c*1) + 0.001)

        precision_c.append(prec)
        recall_c.append(recall)

    return precision_c, recall_c

if __name__ == '__main__':
    eval_model()