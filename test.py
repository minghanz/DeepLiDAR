from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
from submodels import *
from dataloader import preprocess
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = s2dN(1)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

modelpath = os.path.join(ROOT_DIR, args.loadmodel)

if args.loadmodel is not None:
    state_dict = torch.load(modelpath)["state_dict"]
    model.load_state_dict(state_dict)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

processed = preprocess.get_transform(augment=False)

def test(imgL,sparse,mask):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           sparse = torch.FloatTensor(sparse).cuda()
           mask = torch.FloatTensor(mask).cuda()

        imgL= Variable(imgL)
        sparse = Variable(sparse)
        mask = Variable(mask)

        start_time = time.time()
        with torch.no_grad():
            outC, outN, maskC, maskN = model(imgL, sparse, mask)

        tempMask = torch.zeros_like(outC)
        predC = outC[:,0,:,:]
        predN = outN[:,0,:,:]
        tempMask[:, 0, :, :] = maskC
        tempMask[:, 1, :, :] = maskN
        predMask = F.softmax(tempMask)
        predMaskC = predMask[:,0,:,:]
        predMaskN = predMask[:,1,:,:]
        pred1 = predC * predMaskC + predN * predMaskN
        time_temp = (time.time() - start_time)

        output1 = torch.squeeze(pred1)

        return output1.data.cpu().numpy(),time_temp
      
def rmse(gt,img,ratio):
    dif = gt[np.where(gt>ratio)] - img[np.where(gt>ratio)]
    error = np.sqrt(np.mean(dif**2))
    return error
def mae(gt,img,ratio):
    dif = gt[np.where(gt>ratio)] - img[np.where(gt>ratio)]
    error = np.mean(np.fabs(dif))
    return error
def irmse(gt,img,ratio):
    dif = 1.0/gt[np.where(gt>ratio)] - 1.0/img[np.where(gt>ratio)]
    error = np.sqrt(np.mean(dif**2))
    return error
def imae(gt,img,ratio):
    dif = 1.0/gt[np.where(gt>ratio)] - 1.0/img[np.where(gt>ratio)]
    error = np.mean(np.fabs(dif))
    return error

def main_main():
    #    server = 'mcity'
    server = 'name'

    # dataset = 'kitti'
    dataset = 'waymo'
    if dataset == 'kitti':    
        if server == 'mcity':
            data_root_img = '/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data/'
            data_root_dep = data_root_img
            data_root_output = data_root_img
        else:
            data_root_img = '/media/sda1/minghanz/datasets/kitti/kitti_data/'
            data_root_dep = data_root_img
            data_root_output = '/media/sda1/minghanz/datasets/kitti_dep_comp'

        seqs = []
        for date in os.listdir(data_root_dep):
            if os.path.isfile(os.path.join(data_root_dep, date)):
                continue
            for seq in os.listdir(os.path.join(data_root_dep, date)):
                if os.path.isfile(os.path.join(data_root_dep, date, seq)):
                    continue
                seqs.append(os.path.join(date, seq))

    elif dataset == 'waymo':
        if server == 'mcity':
            data_root_img = '/mnt/storage8t/datasets/waymo_kitti/training'
            data_root_dep = data_root_img
            data_root_output = data_root_img
        else:
            data_root_img = '/media/sda1/minghanz/datasets/waymo_kitti'
            data_root_dep = data_root_img
            data_root_output = '/media/sda1/minghanz/datasets/waymo_kitti_dep_comp'

        seqs = []
        for seq in os.listdir(data_root_dep):
            if os.path.isfile(os.path.join(data_root_dep, seq)):
                continue
            seqs.append(seq)

    else:
        raise ValueError('dataset {} not recognized'.format(dataset))
        
    #    print(seqs)
    for seq in seqs:
        print(seq)
        main(data_root_img, data_root_dep, data_root_output, seq, dataset)

def process_single(img_path, lid_path, output_path, dataset):

    imgL_o = skimage.io.imread(img_path)
    imgL_o = np.reshape(imgL_o, [imgL_o.shape[0], imgL_o.shape[1],3])
    
    right_pad = 0
    bottom_pad = 0
    if imgL_o.shape[0] % 2 != 0:
        imgL_o = np.concatenate((imgL_o, imgL_o[[-1]]), axis=0)
        bottom_pad = 1
    if imgL_o.shape[1] % 2 != 0:
        imgL_o = np.concatenate((imgL_o, imgL_o[:, [-1]]), axis=1)
        right_pad = 1
        
    imgL = processed(imgL_o).numpy()
    imgL = np.reshape(imgL, [1, 3, imgL_o.shape[0], imgL_o.shape[1]])

    if dataset == 'kitti':
        #    gtruth = skimage.io.imread(gt_test[inx]).astype(np.float32)
        #    gtruth = gtruth * 1.0 / 256.0
        sparse = skimage.io.imread(lid_path).astype(np.float32)
        sparse = sparse *1.0 / 256.0
    elif dataset == 'waymo':
        sparse = skimage.io.imread(lid_path).astype(np.float32)
        sparse = sparse *1.0 / 256.0
    else:
        raise ValueError('dataset {} not recognized'.format(dataset))

    if bottom_pad == 1:
        sparse = np.concatenate((sparse, sparse[[-1]]), axis=0)
    if right_pad == 1:
        sparse = np.concatenate((sparse, sparse[:, [-1]]), axis=1)

    mask = np.where(sparse > 0.0, 1.0, 0.0)
    mask = np.reshape(mask, [imgL_o.shape[0], imgL_o.shape[1], 1])
    sparse = np.reshape(sparse, [imgL_o.shape[0], imgL_o.shape[1], 1])
    sparse = processed(sparse).numpy()
    sparse = np.reshape(sparse, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])
    mask = processed(mask).numpy()
    mask = np.reshape(mask, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])
    
    output1 = output_path

    pred, time_temp = test(imgL, sparse, mask)
    pred = np.where(pred <= 0.0, 0.9, pred)

    # print(pred.max())
    if bottom_pad == 1:
        pred = pred[:-1]
    if right_pad == 1:
        pred = pred[:, :-1]

    pred_show = pred * 256.0
    pred_show = pred_show.astype('uint16')
    res_buffer = pred_show.tobytes()
    img = Image.new("I",pred_show.T.shape)
    img.frombytes(res_buffer,'raw',"I;16")
    img.save(output1)

    return time_temp

def main(data_root_img, data_root_dep, data_root_output, seq_path, dataset):
    if dataset == 'kitti':
    #    gt_fold = ''
        left_fold = os.path.join(data_root_img , seq_path , 'image_02/data/')
        lidar2_raw = os.path.join(data_root_dep , seq_path , 'proj_depth/velodyne_raw/image_02/')
        dep_output_fold = os.path.join(data_root_output , seq_path , 'depth_completed/DeepLidar/image_02/')
    elif dataset == 'waymo':
        left_fold = os.path.join(data_root_img , seq_path , 'image_00/')
        lidar2_raw = os.path.join(data_root_dep , seq_path , 'depth_00/')
        dep_output_fold = os.path.join(data_root_output , seq_path , 'depth_completed/DeepLidar/depth_00/')
    else:
        raise ValueError("dataset {} not recognized.".format(dataset))

    if os.path.exists(dep_output_fold):
        print(seq_path, 'processed')
        return
    else:
        os.makedirs(dep_output_fold)

    #    gt = [img for img in os.listdir(gt_fold)]
    # image = [img for img in os.listdir(left_fold)]
    lidar2 = [img for img in os.listdir(lidar2_raw)]
    #    gt_test = [gt_fold + img for img in gt]
    # left_test = [os.path.join(left_fold , img) for img in image]
    sparse2_test = [os.path.join(lidar2_raw , img) for img in lidar2]
    #    gt_test.sort()
    # left_test.sort()
    sparse2_test.sort()

    time_all = 0.0

    for inx in range(len(sparse2_test)):
        print("sparse input:", sparse2_test[inx])

        lid_path = sparse2_test[inx]
        img_path = os.path.join(left_fold, sparse2_test[inx].split('/')[-1].split('.')[0]+'.jpg')
        output_path = os.path.join(dep_output_fold , sparse2_test[inx].split('/')[-1])

        time_temp = process_single(img_path, lid_path, output_path, dataset)

        time_all = time_all+time_temp
        print("target path:", output_path)
        print(time_temp)

    print("time: %.8f" % (time_all * 1.0 / 1000.0))

if __name__ == '__main__':
   main_main()

