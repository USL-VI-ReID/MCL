from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
# from data_loader import SYSUData, RegDBData, TestDataOld
# from data_manager import *
# from eval_metrics import eval_sysu, eval_regdb
# from dual_model import embed_net
# from utils import *
from clustercontrast import models
import time 
import matplotlib
from PIL import Image
import pdb
import matplotlib as mpl
import numpy as np
import os
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
mpl.use('Agg')
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import plotly.plotly as py
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from solver import make_optimizer, WarmupMultiStepLR
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import cfg
from clustercontrast import datasets
# from clustercontrast import models
from clustercontrast.model_vit_cmrefine import make_model
from torch import einsum
# from clustercontrast.model_vit_cmrefine.make_model import TransMatcher

from clustercontrast.models.cm import ClusterMemory,ClusterMemory_all,Memory_wise_v3
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_confusionrefine#ClusterContrastTrainer_pretrain_camera_confusionrefine# as ClusterContrastTrainer_pretrain_joint
# from clustercontrast.trainers import ClusterContrastTrainer_pretrain_camera_cpsrefine as ClusterContrastTrainer_pretrain_joint
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_joint# as ClusterContrastTrainer_pretrain_joint_intrac
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine# as ClusterContrastTrainer_pretrain_camera_wise_3
# from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_wise_3_noddcma

from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam,MoreCameraSampler
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
from solver.scheduler_factory import create_scheduler
from typing import Tuple, List, Optional
from torch import Tensor
import numbers
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import cv2

import copy
import os.path as osp
import errno
import shutil

##############
parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
parser.add_argument(
    "--config_file", default="vit_base_ics_288.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
# data
parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                    choices=datasets.names())
parser.add_argument('-b', '--batch-size', type=int, default=2)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--height', type=int, default=288, help="input height")#288 384
parser.add_argument('--width', type=int, default=144, help="input width")#144 128
parser.add_argument('--num-instances', type=int, default=4,
                    help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                            "each identity has num_instances instances, "
                            "default: 0 (NOT USE)")
# cluster
parser.add_argument('--eps', type=float, default=0.6,
                    help="max neighbor distance for DBSCAN")
parser.add_argument('--eps-gap', type=float, default=0.02,
                    help="multi-scale criterion for measuring cluster reliability")
parser.add_argument('--k1', type=int, default=30,#30
                    help="hyperparameter for jaccard distance")
parser.add_argument('--k2', type=int, default=6,
                    help="hyperparameter for jaccard distance")

# model
parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    )
parser.add_argument('--features', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.2,
                    help="update momentum for the hybrid memory")
# optimizer
parser.add_argument('--lr', type=float, default=0.00035,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--iters', type=int, default=400)
parser.add_argument('--step-size', type=int, default=20)
# training configs
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--eval-step', type=int, default=1)
parser.add_argument('--temp', type=float, default=0.05,
                    help="temperature for scaling contrastive loss")
# path
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'data'))
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs'))
parser.add_argument('--pooling-type', type=str, default='gem')
parser.add_argument('--use-hard', action="store_true")
parser.add_argument('--no-cam',  action="store_true")
parser.add_argument('--warmup-step', type=int, default=0)
parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                    help='milestones for the learning rate decay')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='B', help='testing batch size')
parser.add_argument('--low-dim', default=768, type=int,
                    metavar='D', help='feature dimension')
# args = parser.parse_args()

args = parser.parse_args() 
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
dataset='sysu'
dataset = args.dataset
data_path = '/data/modified/'
if dataset == 'sysu':
    data_path = '/data/modified'
    n_class = 395
    log_path = 'sysu_log/'
elif dataset =='regdb':
    data_path = '/data/RegDB/'
    n_class = 206
    log_path = 'regdb_log/'
checkpoint_path = 'logs/sysu_all_res_adc_1.0hmhtnearv3_ice_ins_s1_2_bnn50_all45_trial10_add30_s2clr_Tvnocolor128g2/'

 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 



print('==> Building model..')
# net = create_model(args)
# net.cuda()
# model_path = checkpoint_path + 'model_best.pth.tar'



# checkpoint = load_checkpoint(model_path)

# net.load_state_dict(checkpoint['state_dict'])



if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

cfg.freeze()


net = make_model(cfg, num_class=0, camera_num=0, view_num = 0)

net.cuda()


net = nn.DataParallel(net)#,output_device=1)
#base 18 blma 30

checkpoint = load_checkpoint(osp.join('/data1/adca_prompt_v2/sysu_2p_288_g_s2_baseline_cam_cmav2_biye_noca/1', 'checkpoint.pth.tar'))

net.load_state_dict(checkpoint['state_dict'])



# net.to(device)    
cudnn.benchmark = True

# if len(args.resume)>0:   
#     model_path = checkpoint_path + 'model_best.pth.tar'
#     if os.path.isfile(model_path):
#         print('==> loading checkpoint {}'.format(args.resume))
#         checkpoint = torch.load(model_path)
#         start_epoch = checkpoint['epoch']
#         # pdb.set_trace()
#         net.load_state_dict(checkpoint['state_dict'])
#         print('==> loaded checkpoint {} (epoch {})'
#               .format(args.resume, checkpoint['state_dict']))
#     else:
#         print('==> no checkpoint found at {}'.format(args.resume))


print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288,144)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
dataset='sysu'
if dataset =='sysu':
    # # training set
    
    # testing set
    # test_thermal_list = data_path + 'train_ir.txt'
    # test_color_list = data_path + 'train_rgb.txt'
    test_thermal_list = data_path + 'query_ir_ori.txt'
    test_color_list = data_path + 'gallery_rgb.txt'
    query_img, query_label = load_data(test_thermal_list)
    gall_img, gall_label  = load_data(test_color_list)
    nquery = len(query_label)
    
    cam_id_pos = GenCamIdx(gall_img, gall_label, args.mode)
    
    test_mode = [1, 2]
    
    queryset = TestDataOld(data_path, query_img, query_label, transform = transform_test)

query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

# switch to testing mode
net.eval()
ptr = 0
start = time.time()
print ('Extracting Query Feature...')
query_feat = np.zeros((nquery, args.low_dim))
with torch.no_grad():
    for batch_idx, (input, label ) in enumerate(query_loader):
        batch_num = input.size(0)
        input = input.cuda()
        feat = net(input, input, test_mode[1])
        feat = torch.nn.functional.normalize(feat)
        # pdb.set_trace()
        query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
        ptr = ptr + batch_num
print('Extracting Time:\t {:.3f}'.format(time.time()-start))


def extract_gall_feat(trial_gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((trial_ngall, args.low_dim))
    # gall_feat = np.zeros((ngall, 206))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(trial_gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            feat = net(input, input, test_mode[0])
            feat = torch.nn.functional.normalize(feat)
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat
def frame_image(img, frame_width=10, ptr =1):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    if ptr ==0:
        framed_img[:,:,0] = 255
    elif ptr ==1:
        framed_img[:,:,1] = 255

    framed_img[b:-b, b:-b,:] = img
    return framed_img    
def generate_trial(gall_img, gall_label,cam_id_pos):
    trial_img = []
    trial_label = []
    
    for i in range(len(cam_id_pos)):
        # idx = np.random.choice(cam_id_pos[i],1)
        # idx = idx[0]
        idx = cam_id_pos[i][0]
        trial_img.append(gall_img[idx])
        trial_label.append(gall_label[idx])
    return trial_img, trial_label



all_cmc = 0
all_mAP = 0    


trial_gall_img, trial_gall_label = generate_trial(gall_img, gall_label,cam_id_pos) 
trial_ngall  = len(trial_gall_label)   
trial_gallset  = TestDataOld(data_path, trial_gall_img, trial_gall_label, transform = transform_test)
trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

trial_gall_feat = extract_gall_feat(trial_gall_loader)

distmat = np.matmul(query_feat, np.transpose(trial_gall_feat))

save_path = 'sysu_nn/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
topk = 10
sel_idx = [6,10,17,21,24]#np.random.choice(np.arange(nquery),500)    
i_plot = 0
for i in range(len(sel_idx)):
    print(i_plot)
    # nn_acc = knnAcc[:,i]
    idx = i#sel_idx[i]
    tmp_dist = distmat[idx,:]
    knn_index = np.argpartition(tmp_dist, -topk)[-topk:]
    knn_dist = tmp_dist[knn_index]
    tmp_index = np.argsort(-knn_dist) 
    # pdb.set_trace()
    knn_index = knn_index[tmp_index]
    true_label = query_label[idx]
    trial_gall_label = np.array(trial_gall_label)
    res_label = trial_gall_label[knn_index]
    if (res_label ==true_label).sum()>-1:
        if i_plot%5==0:
           fig = plt.figure(figsize=(64,64))
        
        ptr = i_plot%5
        ax  = plt.subplot(5,topk+1,ptr*(topk+1) + 1)
        query_file = data_path + query_img[idx]
        query_data = Image.open(query_file)
        query_data = query_data.resize((128, 256), Image.ANTIALIAS)
        query_data = np.array(query_data)
        # pdb.set_trace()
        # ax.set_title('query',fontsize=20)
        ax.axis('off')
        query_data = frame_image(query_data, 5, 2)
        plt.imshow(query_data.astype(np.uint8))
        for j in range(topk):                
            s_idx = knn_index[j]
            s_sim = tmp_dist[s_idx]
            ax = plt.subplot(5,topk+1,ptr*(topk+1) + j+2)
            ax.axis('off')
            ax.set_title('%.3f'%(s_sim),fontsize=32)
            # ax.title('{}'.format(s_sim))
            label = res_label[j]
            show_file = data_path + trial_gall_img[s_idx]
            show_img = Image.open(show_file)
            show_img = show_img.resize((128, 256), Image.ANTIALIAS)
            show_img = np.array(show_img)
            # show_img = img_data[s_idx]
            if label == true_label:
                show_img = frame_image(show_img, 5, 1)
                plt.imshow(show_img.astype(np.uint8))
                # plt.subplots_adjust(wspace =0.1, hspace =0.1)
            else:
                show_img = frame_image(show_img, 5, 0)
                plt.imshow(show_img.astype(np.uint8))
            plt.subplots_adjust(left=0.15, right = 0.5,wspace =0.1, hspace =0.2,  bottom =0.15, top =0.5)
        
        if ptr==4:
            plt.savefig(save_path + '{}.pdf'.format(i+100), bbox_inches='tight')
            # pdb.set_trace()
            plt.close()
        i_plot = i_plot +1
# cmc, mAP = compute_accuracy(-distmat, query_label, trial_gall_label,topk=20)




