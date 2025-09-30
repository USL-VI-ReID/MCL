from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
from clustercontrast.models.cm import ClusterMemory
import numpy as np
import math

from clustercontrast.utils.batch_norm import get_norm
from clustercontrast.utils.loss import cross_modality_label_preserving_loss
from clustercontrast.utils.weight_init import weights_init_kaiming
#import collections


part=5
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):

        # nearest_rgb_ir = targets.max(dim=1, keepdim=True)[0]
        # mask_neighbor_rgb_ir = torch.gt(targets, nearest_rgb_ir * 0.8)
        # num_neighbor_rgb_ir = mask_neighbor_rgb_ir.sum(dim=1)+1
        # print(num_neighbor_rgb_ir)
        log_probs = self.logsoftmax(inputs)
        loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).mean(0).sum()
        # # loss = (- (targets).detach() * log_probs).mul(mask_neighbor_rgb_ir).sum(dim=1)#.mean(0).sum()#.mean()
        # # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).mul(mask_neighbor_rgb_ir).sum(dim=1)#.mean(0).sum()#.mean()
        # loss = loss.div(num_neighbor_rgb_ir).mean()

        # sim_rgb_ir=inputs
        # sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
        # nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
        # nearest_rgb_ir_2 = targets.max(dim=1, keepdim=True)[0]
        # mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
        # mask_neighbor_rgb_ir_2 = torch.gt(targets, nearest_rgb_ir_2 * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps

        # num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_rgb_ir_2).sum(dim=1)+1
        # score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
        # loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_rgb_ir_2).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
        # loss = loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##



        # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).sum(1)
        # loss = loss.div(num_neighbor_rgb_ir).mean()
        # print(loss.item())
        return loss

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        self.ratio = 0.2

    def forward(self, pred, target):

        B = pred.shape[0]
        pred = torch.softmax(pred, dim=1)
        target = torch.softmax(target / self.ratio, dim=1)

        loss = (-pred.log() * target).sum(1).sum() / B

        return loss
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=True):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        # correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss#, correct

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss




class ClusterContrastTrainer_pretrain_camera_confusionrefine_clusterattv2(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_confusionrefine_clusterattv2, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach().t()
        # # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_ir, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir).detach().t()
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # percam_memory_all=percam_memory_rgb+percam_memory_ir
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_all]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_all]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_all, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb)
        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_all, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir)
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

        # del percam_memory_ir,percam_memory_rgb
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            # cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()



            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
                

            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(f_out_ir_r, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0
################cpsrefine
            # f_agg_ir = self.encoder.module.b2(f_out_ir.unsqueeze(0).detach(),lab=labels_ir, cluster_num =self.memory_ir.features.size(0))
            # f_agg_rgb = self.encoder.module.b2(f_out_rgb.unsqueeze(0).detach(),lab=labels_rgb, cluster_num =self.memory_rgb.features.size(0))
            # f_agg_all = torch.cat((f_agg_rgb,f_agg_ir),dim=1)
            # f_agg_all = self.encoder.module.b2_norm(f_agg_all)
            # f_agg_rgb = f_agg_all[:,:f_out_rgb.size(0)].squeeze(0)
            # f_agg_ir = f_agg_all[:,f_out_rgb.size(0):].squeeze(0)




            # loss_ir_agg = self.memory_ir(f_agg_ir, labels_ir,training_momentum=1)# + self.memory_ir(f_out_ir_r, labels_ir)
            # loss_rgb_agg = self.memory_rgb(f_agg_rgb, labels_rgb,training_momentum=1)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            # outputs = f_agg_ir.mm(self.memory_ir.features.t())
            # outputs /= self.temp
            # loss = F.cross_entropy(outputs, targets)
            # if epoch>=0:#self.cmlabel:


            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            loss = loss_ir+loss_rgb#+loss_ir_agg+loss_rgb_agg#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            # loss_camera_rgb_log.update(loss_camera_rgb.item())
            # loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            # loss_ins_ir_log.update(loss_ir_agg.item())
            # loss_ins_rgb_log.update(loss_rgb_agg.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'Loss agg ir {:.3f} ({:.3f})\t'
                      'Loss agg rgb {:.3f} ({:.3f})\t'
                    #   'camera ir {:.3f} ({:.3f})\t'
                    #   'camera rgb {:.3f} ({:.3f})\t'
                    #   'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'ir_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,
                              loss_ins_ir_log.val,loss_ins_ir_log.avg,loss_ins_rgb_log.val,loss_ins_rgb_log.avg))
                            #   ,\
                            #   loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                            #   ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                            #   ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item(),flush=True)
            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f}\t'
            #           'Loss rgb {:.3f}\t'
            #           'camera ir {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
            #     print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            #     print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            #     print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_query_ir_p,',loss_query_ir_p.item())
                # print('loss_query_ir_n,',loss_query_ir_n.item())
                # print('loss_query_rgb_p,',loss_query_rgb_p.item())
                # print('loss_query_rgb_n,',loss_query_rgb_n.item())
                # print('score_log_ir',score_log_ir)
                # print('score_log_rgb',score_log_rgb)
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score
@torch.no_grad()
def label_sort(label):
    all_label=[]
    lable_his=[]
    n=-1
    for l_i in label:
        if l_i == -1:
            all_label.append(l_i)
        elif l_i in lable_his:
            all_label.append(n)
        else:
            n=n+1
            all_label.append(n)
            lable_his.append(l_i)
    all_label=np.asarray(all_label)
    return torch.from_numpy(all_label).view(-1).cuda()

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers.cuda()



class dy_cc(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2):
        super(dy_cc, self).__init__()


        self.temp = temp


    def forward(self, inputs, targets):
        # inputs: B*2048, features: L*2048

        inputs = F.normalize(inputs, dim=1).mm(F.normalize(inputs, dim=1).t())
        inputs /= self.temp
        B = inputs.size(0)
        num_samples = B
        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = targets
        labels = targets

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)


class MultiSimilarityLoss(nn.Module):
    def __init__(self, thresh=0.5, margin=0.1, scale_pos=2, scale_neg=40):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2
        self.scale_neg = 40

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        # sim_mat = torch.matmul(feats, torch.t(feats))



        sim_mat = cosine_dist(feats, feats)

        epsilon = 1e-5
        loss = list()


        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(pos_pair_) == 0:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], device='cuda:0', requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    # return 1-cosine
    return cosine


def sinkhorn_solver(P, lambda_sk=25, max_iter=500):
    
    tt = time.time()
    P =P.cpu()
    num_instance = P.shape[0]
    num_clusters = P.shape[1]
    
    alpha = torch.ones((num_instance, 1)) / num_instance  # initial value for alpha
    beta = torch.ones((num_clusters, 1)) / num_clusters  # initial value for beta
    
    inv_K = 1. / num_instance
    inv_N = 1. / num_clusters   
    PS = torch.exp(-lambda_sk * P)
        
    err = 1e6
    step = 0  
    
    while err > 1e-1 and step < max_iter:
        alpha = inv_K / (torch.mm(PS, beta))  # (KxN) @ (N,1) = K x 1
        beta_new = inv_N / (torch.mm(alpha.t(), PS)).T  # ((1,K) @ (KxN)).t() = N x 1
        if step % 10 == 0:
            err = np.nansum(np.abs(beta.detach().cpu()/ beta_new.detach().cpu() - 1))
        beta = beta_new
        step += 1
        
    # print("Sinkhorn-Knopp   Error: {:.3f}   Total step: {}   Total time: {:.3f}".format(err, step, time.time() - tt))
    P_out = torch.diag(alpha.squeeze()).mm(PS).mm(torch.diag(beta.squeeze()))
        
    return P_out

def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1).cpu()
    y = y.view(n, -1).cpu()
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat


class ClusterContrastTrainer_pretrain_camera_distill(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_distill, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft =  SoftEntropy().cuda()# SoftEntropy().cuda() # SoftCrossEntropyLoss().cuda()#

        self.criterion_ms = MultiSimilarityLoss().cuda()


    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1


        part=4
        # c_out_rgb_ir =  sinkhorn_solver(pairwise_distance(self.wise_memory_rgb.features, self.wise_memory_ir.features)).cuda()
        # c_out_ir_rgb = sinkhorn_solver(pairwise_distance(self.wise_memory_ir.features, self.wise_memory_rgb.features)).cuda()
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)



            # indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            # indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(f_out_ir_r, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            ld_rgb_ir = torch.tensor([0.]).cuda()
            ld_ir_rgb = torch.tensor([0.]).cuda()

            ld_rgb_rgb = torch.tensor([0.]).cuda()
            ld_ir_ir = torch.tensor([0.]).cuda()
            thresh=0.9
            hm_thresh=0.9
            lamda_d_neibor=0.5
            # loss_ir_ms = self.criterion_ms(f_out_ir, labels_ir)
            # loss_rgb_ms = self.criterion_ms(f_out_rgb, labels_rgb)
            # if epoch>=0:

#                 # p_out_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.t()) # 1 Cir
#                 # p_out_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.t()) # 1 Crgb
#                 # # c_out_rgb_ir =  sinkhorn_solver(pairwise_distance(self.wise_memory_rgb.features, self.wise_memory_ir.features)).cuda()#self.memory_rgb.features.mm(self.memory_ir.features.t())# Crgb Cir
#                 # # c_out_rgb_ir =  self.wise_memory_rgb.features.mm(self.wise_memory_ir.features.t())
#                 # c_out_cas_rgb_ir =  p_out_rgb.mm(c_out_rgb_ir) 
#                 # p_out_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.t()) # 1 Crgb
#                 # p_out_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.t()) # 1 Cir
#                 # # c_out_ir_rgb = self.wise_memory_ir.features.mm(self.wise_memory_rgb.features.t())#self.memory_ir.features.mm(self.memory_rgb.features.t())#  Cir Crgb
#                 # # c_out_ir_rgb = sinkhorn_solver(pairwise_distance(self.wise_memory_ir.features, self.wise_memory_rgb.features)).cuda()#self.memory_ir.features.mm(self.memory_rgb.features.t())#  Cir Crgb
                
#                 # c_out_cas_ir_rgb =  p_out_ir.mm(c_out_ir_rgb) 
#                 # c_out_cas_rgb_rgb =  p_out_rgb_ir.mm(c_out_ir_rgb)  #p_out_rgb_ir.mm(c_out_ir_rgb)
#                 # c_out_cas_ir_ir =  p_out_ir_rgb.mm(c_out_rgb_ir)   #p_out_ir_rgb.mm(c_out_rgb_ir)
#                 # if epoch %2 == 0:
#                 #     sim_prob_rgb_ir = c_out_cas_rgb_ir
#                 #     sim_rgb_ir = p_out_rgb_ir
#                 #     nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
#                 #     nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
#                 #     mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 #     mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 #     num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1
#                 #     sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
#                 #     sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
#                 #     score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                 #     # print('score_intra',score_intra)
#                 #     score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
#                 #     # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
#                 #     rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                 #     ld_rgb_ir = lamda_d_neibor*rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(mask_neighbor_intra_soft) ##.mul(rgb_ir_ca)

#                 # else:
#                 #     sim_prob_ir_rgb = c_out_cas_ir_rgb
#                 #     sim_ir_rgb = p_out_ir_rgb
#                 #     nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
#                 #     nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
#                 #     mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 #     mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 #     num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_prob_ir_rgb).sum(dim=1)+1
#                 #     sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
#                 #     sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
#                 #     score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                 #     # print('score_intra',score_intra)
#                 #     score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
#                 #     # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
#                 #     ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_prob_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                 #     ld_ir_rgb = lamda_d_neibor*ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(ir_rgb_ca).mul(ir_rgb__ca)mul(rgb_ir_ca).mul(mask_neighbor_intra_soft) ##.mul(rgb_ir_ca)

#                 # sim_prob_rgb_rgb = c_out_cas_rgb_rgb
#                 # sim_rgb_rgb = p_out_rgb
#                 # nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
#                 # nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
#                 # mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * hm_thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 # mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * hm_thresh).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 # num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
#                 # # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
#                 # # sim_prob_rgb_rgb = F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())/0.05,dim=1)#B N
#                 # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())/0.05,dim=1)
#                 # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
#                 # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                 # # print('score_intra',score_intra)
#                 # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
#                 # # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
#                 # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1) #.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                 # ld_rgb_rgb = lamda_d_neibor*rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(rgb_ca).mul(rgb_ca).mul(rgb_ca)..mul(rgb_ca)mul(mask_neighbor_intra_soft) ##
                



#                 # sim_prob_ir_ir = c_out_cas_ir_ir
#                 # sim_ir_ir = p_out_ir
#                 # nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
#                 # nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
#                 # mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * hm_thresh).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 # mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * hm_thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                 # num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
#                 # # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
#                 # # sim_prob_ir_ir = F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())#F.softmax(F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
#                 # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
#                 # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
#                 # score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                 # # print('score_intra',score_intra)
#                 # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
#                 # # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
#                 # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                 # ld_ir_ir = lamda_d_neibor*ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(ir_ca).mul(ir_ca).mul(ir_ca).mul(ir_ca).mul(mask_neighbor_intra_soft) ##


# # #######################
#                 # if epoch %2 == 0:
#                     # ld_rgb_ir =   0.1*self.criterion_ce_soft(p_out_rgb_ir/0.05, c_out_cas_rgb_ir/0.01)#(1. / (1. + math.exp(loss_rgb_log.avg) ))
#                 #     # ld_ir_ir = 0.5*self.criterion_kl(p_out_ir/0.05, c_out_cas_ir_ir/0.01)#(1. / (1. + math.exp(loss_ir_log.avg) ))
#                 # else:
#                 #     # ld_rgb_rgb = 0.5*self.criterion_kl(p_out_rgb/0.05, c_out_cas_rgb_rgb/0.01)#(1. / (1. + math.exp(loss_rgb_log.avg) ))
#                 #     ld_ir_rgb =   0.1* self.criterion_ce_soft(p_out_ir_rgb/0.05, c_out_cas_ir_rgb/0.01)#criterion_kl
#                 # # ld_rgb_ir = (1. / (1. + math.exp(loss_ir_log.avg)) )*self.criterion_ce_soft(p_out_rgb_ir, c_out_cas_rgb_ir)#
#                 # # ld_ir_rgb=(1. / (1. + math.exp(loss_rgb_log.avg) )) * self.criterion_ce_soft(p_out_ir_rgb, c_out_cas_ir_rgb)



# ########v0
#                 p_out_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.memory_ir.features.t())
#                 p_out_rgb = F.normalize(f_out_rgb, dim=1).mm(self.memory_rgb.features.t())
#                 c_out_rgb_ir = sinkhorn_solver(pairwise_distance(self.memory_rgb.features, self.memory_ir.features)).cuda()
#                 c_out_cas_rgb_ir =  p_out_rgb.mm(c_out_rgb_ir) 

#                 p_out_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.memory_rgb.features.t()) # B 768 768 C  B C C 768
#                 p_out_ir = F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.t())
#                 c_out_ir_rgb = sinkhorn_solver(pairwise_distance(self.memory_ir.features, self.memory_rgb.features)).cuda()
#                 c_out_cas_ir_rgb =  p_out_ir.mm(c_out_ir_rgb) 

#                 # p_out_rgb_ir = f_out_rgb.mm(self.memory_ir.features.t())
#                 # # p_out_rgb = f_out_rgb.mm(self.memory_rgb.features.t())

#                 # # p_out_ir_rgb = f_out_ir.mm(self.memory_rgb.features.t()) # B 768 768 C  B C C 768
#                 # # p_out_ir = f_out_ir.mm(self.memory_ir.features.t())

#                 # rgb_softmax = F.softmax(p_out_rgb, dim=1)
#                 # rgb_ir_softmax = F.softmax(p_out_rgb_ir, dim=1)
#                 # ir_softmax = F.softmax(p_out_ir, dim=1)
#                 # ir_rgb_softmax = F.softmax(p_out_ir_rgb, dim=1)

#                 # aug_ir_rgb = f_out_ir+ir_rgb_softmax.mm(self.memory_rgb.features)
#                 # p_aug_ir_rgb = F.normalize(aug_ir_rgb, dim=1).mm(self.memory_ir.features.t()) 
#                 # # p_aug_ir_rgb = aug_ir_rgb.mm(self.memory_ir.features.t()) 
#                 # # aug_ir_rgb_softmax = F.softmax(p_aug_ir_rgb, dim=1)
#                 # # ld_ir_rgb = 0.5*self.criterion_ce_soft(f_out_ir, aug_ir_rgb)
#                 ld_ir_rgb = 1*self.criterion_ce_soft(p_out_ir_rgb/0.05, c_out_cas_ir_rgb/0.01)#0.5 51.63

#                 # aug_rgb_ir = f_out_rgb+rgb_ir_softmax.mm(self.memory_ir.features)
#                 # # p_aug_rgb_ir = aug_rgb_ir.mm(self.memory_rgb.features.t()) 
#                 # p_aug_rgb_ir = F.normalize(aug_rgb_ir, dim=1).mm(self.memory_rgb.features.t()) 
#                 # # aug_rgb_ir_softmax = F.softmax(p_aug_rgb_ir, dim=1)
#                 ld_rgb_ir = 1*self.criterion_ce_soft(p_out_rgb_ir/0.05, c_out_cas_rgb_ir/0.01)
#                 # ld_rgb_ir = 0.5*self.criterion_ce_soft(p_out_rgb, p_aug_rgb_ir)#

#                 # # aug_rgb_ir = p_out_rgb_ir.mm(self.memory_ir.features)
#                 # # p_aug_rgb_ir = F.normalize(aug_rgb_ir, dim=1).mm(self.memory_rgb.features.t()) 
#                 # # ld_rgb_ir = self.criterion_ce_soft(p_out_rgb, p_aug_rgb_ir)



            # transfer_loss = -torch.norm(rgb_softmax_1,'nuc')/rgb_softmax_1.shape[0] + \
            #                         -torch.norm(rgb_softmax_2,'nuc')/rgb_softmax_2.shape[0] + \
            #                         -torch.norm(ir_softmax_3,'nuc')/ir_softmax_3.shape[0] + \
            #                         -torch.norm(ir_softmax_4,'nuc')/ir_softmax_4.shape[0]



            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0


            lamda_i = 1
####################
            # loss_ins_ir = 0*self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= 0*self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#

            loss =loss_ir+loss_rgb#+ ld_ir_rgb+ld_rgb_ir#+ld_rgb_rgb+ld_ir_ir#+ +transfer_loss#+loss_ir_ms+loss_rgb_ms#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())

            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())


            rgb_rgb_loss_log.update(ld_rgb_rgb.item())
            ir_ir_loss_log.update(ld_ir_ir.item())

            loss_ins_ir_log.update(ld_rgb_ir.item())
            loss_ins_rgb_log.update(ld_ir_rgb.item())


            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'ld_ir_rgb {:.3f} ({:.3f})\t'
                        'ld_rgb_ir {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                    #   'camera ir {:.3f} ({:.3f})\t'
                    #   'camera rgb {:.3f} ({:.3f})\t'
                    #   'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'ir_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,
                              loss_ins_rgb_log.val,loss_ins_rgb_log.avg, loss_ins_ir_log.val,loss_ins_ir_log.avg,
                              ir_ir_loss_log.val,ir_ir_loss_log.avg, rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg
                              ))
                            #   ,\
                            #   loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                            #   ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                            #   ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item(),flush=True)

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score




class ClusterContrastTrainer_pretrain_camera_confusionrefine(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_confusionrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_ir2rgb = memory
        self.memory_rgb2ir = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.dycc=dy_cc().cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()
        
        #self.bottleneck = get_norm('BN', 768, bias_freeze=True)
        # modality-specific batch normalization
        #self.bottleneck_modality = nn.ModuleList(torch.nn.BatchNorm2d(768) for _ in range(self.num_modalities))
        #self.bottleneck.cuda()
        #self.bottleneck = nn.DataParallel(self.bottleneck)
        #self.bottleneck.apply(weights_init_kaiming)
        #for bn in self.bottleneck_modality:
        #  bn.cuda()
        #  bn = nn.DataParallel(bn)
        #  bn.apply(weights_init_kaiming)
        
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        
        loss_LPL_ir_log = AverageMeter()
        loss_LPL_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        

        end = time.time()
        for i in range(train_iters):
            
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)
            
            loss_LPL_rgb = torch.tensor([0.]).cuda()
            loss_LPL_ir = torch.tensor([0.]).cuda()


            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()

            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()

            
            f_out,f_out_rgb,f_out_ir,f_out_rgb_llm,f_out_ir_llm,labels_rgb,labels_ir,\
                  cid_rgb,cid_ir,index_rgb,index_ir,rgb2rgb_feat,rgb2ir_feat,ir2ir_feat,ir2rgb_feat = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                  cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            
                  
            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb) 


            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()

            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0

            lamda_i = 1

            loss = loss_ir+loss_rgb
            #+llmkl_loss_rgb+llmkl_loss_ir#+llmtri_loss_rgb+llmtri_loss_ir#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            # loss_camera_rgb_log.update(loss_camera_rgb.item())
            # loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())
            
            loss_LPL_rgb_log.update(loss_LPL_rgb.item())
            loss_LPL_ir_log.update(loss_LPL_ir.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg))                       
    def train2(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_ir2rgb_log = AverageMeter()
        loss_rgb2ir_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        
        loss_CLP_ir_log = AverageMeter()
        loss_CLP_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        

        end = time.time()
        for i in range(train_iters):
            
            feature_dict_out = collections.defaultdict(list)
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)


            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()

            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()

            
            f_out,f_out_rgb,f_out_ir,f_out_rgb_llm,f_out_ir_llm,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir,rgb2rgb_feat,rgb2ir_feat,ir2ir_feat,ir2rgb_feat = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            
            loss_CLP_rgb = torch.tensor([0.]).cuda()
            loss_CLP_ir = torch.tensor([0.]).cuda()
            loss_ir2rgb = torch.tensor([0.]).cuda()
            loss_rgb2ir = torch.tensor([0.]).cuda()
              
            # global feature bn
            #f_out_all = torch.cat((f_out_rgb,f_out_ir),0) # f_out_all [256+128, 768]
            #bn_feat = self.bottleneck(f_out_all.unsqueeze(-1).unsqueeze(-1))
            #bn_feat = bn_feat[..., 0, 0]
            #feature_rgb_per = f_out_rgb.unsqueeze(-1).unsqueeze(-1)
            #feature_ir_per = f_out_ir.unsqueeze(-1).unsqueeze(-1)
            
            # 0 -> f_out_rgb
            # 1 -> f_out_ir
            # rgb
            #for i in range(self.num_modalities):
            #    if i == int(0):
            #        feature_dict_out[int(0)].append(self.bottleneck_modality[i](feature_rgb_per)[..., 0, 0])
            #    else:
            #        self.bottleneck_modality[i].eval()
            #        feature_dict_out[int(0)].append(self.bottleneck_modality[i](feature_rgb_per)[..., 0, 0])
            #        self.bottleneck_modality[i].train()
            # ir        
            #for i in range(self.num_modalities):
            #    if i == int(1):
            #        # self.bottleneck_modalityΪir
            #        feature_dict_out[int(1)].append(self.bottleneck_modality[i](feature_ir_per)[..., 0, 0])
            #    else:
            #        self.bottleneck_modality[i].eval()
            #        # self.bottleneck_modalityΪrgb
            #        feature_dict_out[int(1)].append(self.bottleneck_modality[i](feature_ir_per)[..., 0, 0])
            #        self.bottleneck_modality[i].train()
            
            
            lamda_m = math.log(epoch * i / train_iters + 1.).
            
            #current_iter = i + train_iters * epoch
            #total_iter = train_iters * 50
            
            
            loss_CLP_ir = label_preserving_loss(f_out_ir, ir2rgb_feat, labels_ir)
            loss_CLP_rgb = label_preserving_loss(f_out_rgb, rgb2ir_feat, labels_rgb)
            
            #loss_ir = self.memory_ir(f_out_ir, labels_ir) #+ 0.001 * self.memory_ir(f_out_ir_support, labels_ir)
            #loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb) #+ 0.001 * self.memory_rgb(f_out_rgb_support, labels_rgb)
            #loss_ir = self.memory_ir(f_out_ir, labels_ir) + 2.5 * self.memory_ir(ir2rgb_feat, labels_ir)
            #loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb) + 2.5 * self.memory_rgb(rgb2ir_feat, labels_rgb)
            
            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(ir2rgb_feat, labels_ir)# + self.memory_ir2rgb(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# + self.memory_rgb(rgb2ir_feat, labels_rgb)# + self.memory_rgb2ir(f_out_rgb, labels_rgb)
            
            #loss_ir = self.memory_ir(f_out_ir, labels_ir) + self.memory_ir2rgb(f_out_ir, labels_ir)
            #loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb) + self.memory_rgb2ir(f_out_rgb, labels_rgb)
            
            loss_ir2rgb = self.memory_ir2rgb(f_out_ir, labels_ir)
            loss_rgb2ir = self.memory_rgb2ir(f_out_rgb, labels_rgb)
            #loss_ir = self.memory_ir(f_out_ir, labels_ir) # + self.memory_ir(f_out_ir_llm, labels_ir)
            #loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_llm, labels_rgb)


            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0

            lamda_i = 1
            
            #loss = loss_ir+loss_rgb
            loss = loss_ir+loss_rgb+loss_rgb2ir+loss_ir2rgb+loss_CLP_ir+loss_CLP_rgb
            #loss = loss_ir+loss_rgb+loss_GSL#+llmkl_loss_rgb+llmkl_loss_ir#+llmtri_loss_rgb+llmtri_loss_ir#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            loss_ir2rgb_log.update(loss_ir2rgb.item())
            loss_rgb2ir_log.update(loss_rgb2ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())
            
            loss_CLP_rgb_log.update(loss_LPL_rgb.item())
            loss_CLP_ir_log.update(loss_LPL_ir.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        'Loss {:.3f} ({:.3f})\t'
                        'Loss ir {:.3f} ({:.3f})\t'
                        'Loss rgb {:.3f} ({:.3f})\t'
                        'Loss CLP ir {:.3f} ({:.3f})\t'
                        'Loss CLP rgb {:.3f} ({:.3f})\t'
                        'Loss ir2rgb {:.3f} ({:.3f})\t'
                        'Loss rgb2ir {:.3f} ({:.3f})\t'
                        .format(epoch, i + 1, len(data_loader_rgb),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg,
                                losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,loss_CLP_ir_log.val,loss_CLP_ir_log.avg,loss_CLP_rgb_log.val,loss_CLP_rgb_log.avg,loss_ir2rgb_log.val,loss_ir2rgb_log.avg,loss_rgb2ir_log.val,loss_rgb2ir_log.avg))
                              
                              
                              
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score




class ClusterContrastTrainer_pretrain_prompt(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_prompt, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.dycc=dy_cc().cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()
        self.alpha = 0.999
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4


        # del percam_memory_ir,percam_memory_rgb
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)


            inputs_ir,inputs_ir2,inputs_ir3,inputs_ir4,inputs_ir5,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1,inputs_rgb2,inputs_rgb3,inputs_rgb4,inputs_rgb5, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.encoder.module.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.encoder.module.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)

            inputs_rgb_prompt = torch.cat((inputs_rgb2,inputs_rgb3,inputs_rgb4,inputs_rgb5),1)
            inputs_ir_prompt = torch.cat((inputs_ir2,inputs_ir3,inputs_ir4,inputs_ir5),1)

            prompt_use=True
            if epoch>=10:
                prompt_use=True
            _,f_out_rgb,f_out_ir,f_out_rgb_llm,f_out_ir_llm,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir,prompt_use=prompt_use,inputs_rgb_prompt=inputs_rgb_prompt,inputs_ir_prompt=inputs_ir_prompt)

            
            loss_ir = self.memory_ir(f_out_ir, labels_ir) # + self.memory_ir(f_out_ir_llm, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_llm, labels_rgb)


            # llmkl_loss_rgb = self.criterion_kl(f_out_rgb, Variable(f_out_rgb_llm))#self.memory_ir(f_out_ir_llm, labels_ir )#
            # llmkl_loss_ir = self.criterion_kl(f_out_ir, Variable(f_out_ir_llm))#self.memory_rgb( f_out_rgb_llm, labels_rgb)#

            # llmtri_loss_rgb = self.tri(torch.cat((f_out_rgb,f_out_rgb_llm),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            # llmtri_loss_ir = self.tri(torch.cat((f_out_ir,f_out_ir_llm),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))




            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0

            lamda_i = 1

            loss = loss_ir+loss_rgb#+llmkl_loss_rgb+llmkl_loss_ir#+llmtri_loss_rgb+llmtri_loss_ir#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.encoder, self.encoder.module.prompt_encoder, self.alpha, epoch*len(data_loader_rgb)+i)
# 
            with torch.no_grad():
                self.encoder.module.wise_memory_ir.updateEM(f_out_ir, index_ir)
                self.encoder.module.wise_memory_rgb.updateEM(f_out_rgb, index_rgb)

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            # loss_camera_rgb_log.update(loss_camera_rgb.item())
            # loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                    #   'camera ir {:.3f} ({:.3f})\t'
                    #   'camera rgb {:.3f} ({:.3f})\t'
                    #   'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'ir_ir_loss_log {:.3f} ({:.3f})\t'
                    #   'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, img2,img3,img4,img5,name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), img2.cuda(),img3.cuda(),img4.cuda(),img5.cuda(),pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, img2,img3,img4,img5,name, pids, cids, indexes = inputs
        return imgs.cuda(), img2.cuda(),img3.cuda(),img4.cuda(),img5.cuda(),pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None,prompt_use=False,inputs_rgb_prompt=None,inputs_ir_prompt=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir,prompt_use=prompt_use,inputs_rgb_prompt=inputs_rgb_prompt,inputs_ir_prompt=inputs_ir_prompt)

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        #     if 'patch_embed' or "patch_embed2" in ema_param[0].split('.'):
        #         continue
        #     else:
        #         ema_param[1].data.mul_(alpha).add_(1 - alpha, param[1].data)

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        # for name, param in self.prompt_encoder.named_parameters():
        #     # param.requires_grad_(False)
        #     # if 'adapter' in name.split('.'):
        #     #     print(name)
        #     #     param.requires_grad_(True)
        #     if 'patch_embed' in name.split('.'):
        #         print(name)
        #         param.requires_grad_(True)
        #     elif 'patch_embed2' in name.split('.'):
        #         print(name)
        #         param.requires_grad_(True)
        #     else:
        #         param.requires_grad_(False)


    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score


class ClusterContrastTrainer_pretrain_camera_confusionrefine_noice(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_confusionrefine_noice, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.0#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach().t()
        # # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_ir, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir).detach().t()
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # percam_memory_all=percam_memory_rgb+percam_memory_ir
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_all]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_all]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_all, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb)
        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_all, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir)
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

        # del percam_memory_ir,percam_memory_rgb
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            # cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()



            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
                


            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(f_out_ir_r, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0


            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            loss = loss_ir+loss_rgb#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            loss_camera_rgb_log.update(loss_camera_rgb.item())
            loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item(),flush=True)
            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f}\t'
            #           'Loss rgb {:.3f}\t'
            #           'camera ir {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
            #     print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            #     print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            #     print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_query_ir_p,',loss_query_ir_p.item())
                # print('loss_query_ir_n,',loss_query_ir_n.item())
                # print('loss_query_rgb_p,',loss_query_rgb_p.item())
                # print('loss_query_rgb_n,',loss_query_rgb_n.item())
                # print('score_log_ir',score_log_ir)
                # print('score_log_rgb',score_log_rgb)
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_all =  memory
        self.wise_memory_rgb =  memory
        self.wise_memory_ir =  memory

        self.nameMap_rgb = []
        self.nameMap_ir = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)
        # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
        self.camstart=0
        self.tri = TripletLoss_WRT()
        self.criterion_kl = KLDivLoss()
        self.criterion_ce_soft = SoftEntropy().cuda()
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,all_label=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()

        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        
        
        # # percam_memory_rgb = [self.wise_memory_rgb.features, self.wise_memory_ir.features]
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        distribute_map_ir=percam_memory_rgb[1]
        distribute_map_rgb=percam_memory_rgb[0]
        # # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=rgb_softmax_dim


        memory_dy_ir = ClusterMemory(self.encoder.module.in_planes*part, distribute_map_ir.size(0), temp=0.05,
                               momentum=0.5, use_hard=False).cuda()
        # memory_dy_rgb = ClusterMemory(self.encoder.module.in_planes*part, distribute_map_rgb.size(0), temp=0.05,
        #                        momentum=0.5, use_hard=False).cuda()

        memory_dy_ir.features = distribute_map_ir#.cuda()
        # memory_dy_rgb.features = distribute_map_ir#.cuda()

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach()#.t()


        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_rgb.cuda())

        # Note=open('x.txt',mode='a+')

        start_cam=0
        ir_num = len(all_label_ir)
        rgb_num = len(all_label)-ir_num
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            # indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            # indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = []#torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # # for path,cameraid in  zip(name_ir,cids_ir):
            # #     print(path,cameraid)

            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()




            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            # indexes_all = torch.cat((index_rgb,index_ir),-1)
            cid_all=torch.cat((cid_rgb,cid_ir),-1)

#####################################
            labels_all = torch.cat((labels_rgb,labels_ir),-1)
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
            lamda_c=0.1
            # start=30
            loss_camera_ir=torch.tensor([0.]).cuda()
            loss_camera_rgb=torch.tensor([0.]).cuda()
            loss_camera_all = torch.tensor([0.]).cuda()
            rgb_rgb_loss = torch.tensor([0.]).cuda()
            ir_ir_loss = torch.tensor([0.]).cuda()
            loss_confusion_all  = torch.tensor([0.]).cuda()
            # if epoch >= self.camstart:
            loss_camera_all = self.camera_loss(f_out_all,cid_all,labels_all,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb,cross_m=True)#self.camera_loss(f_out_ir,cid_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
                # loss_camera_rgb = self.camera_loss(f_out_rgb,cid_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)

            # concate_mem = torch.cat(percam_memory_rgb,dim=0)  #C_RGB+C_IR dim
            # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            # sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_all, dim=1).mm(percam_memory_rgb[i].detach().data.t()),dim=1) for i in range(len(percam_memory_rgb))],dim=1)
            # # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
            # # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1)
            # confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
            # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
            # # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
            # # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
            # loss_confusion_all = self.tri(torch.cat((f_out_all,confusion_feat_rgb),dim=0),torch.cat((labels_all,labels_all),dim=-1))
            # # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            if epoch>=700:
                # features_rgb = F.normalize(f_out_rgb)#features_rgb_.cuda()#
                # features_rgb_input = F.normalize(f_out_all,dim=1).mm(distribute_map_rgb)# self.encoder.module.classifier_rgb()#*20
                # features_rgb= features_rgb_input*20
                # # rgb_softmax_dim= self.encoder.module.rgb_softmax_dim
                # features_rgb_1 = F.softmax(features_rgb[:,:rgb_softmax_dim[0]], dim=1)
                # features_rgb_2 = F.softmax(features_rgb[:,rgb_softmax_dim[0]:], dim=1)
                # features_rgb_sim = torch.cat((features_rgb_1,features_rgb_2), dim=1)
                # rgb_rgb_loss = self.criterion_ce_soft(features_rgb_input,features_rgb_sim)

                # if epoch%2==0:
                # for l in range(part):
                #     if l ==0:
                #         ins_sim_rgb_all = F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #     else:
                #         ins_sim_rgb_all += F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                # ins_sim_rgb_sim = ins_sim_rgb_all*20#F.normalize(f_out_all,dim=1).mm(distribute_map_rgb)# self.encoder.module.classifier_rgb()#*20
                # # features_rgb= features_rgb_input*20
                # features_rgb_1 = F.softmax(ins_sim_rgb_sim[:,:rgb_softmax_dim[0]], dim=1)
                # features_rgb_2 = F.softmax(ins_sim_rgb_sim[:,rgb_softmax_dim[0]:], dim=1)
                # features_rgb_sim = torch.cat((features_rgb_1,features_rgb_2), dim=1)
                # features_rgb_input = F.normalize(f_out_all,dim=1).mm(distribute_map_rgb.t()) #ins_sim_rgb_all#best: 26.9% *
                # rgb_rgb_loss = self.criterion_ce_soft(features_rgb_input,features_rgb_sim)

                # for l in range(part):

                #     if l ==0:
                #         ins_sim_rgb_all = F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #     else:
                #         ins_sim_rgb_all += F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                # if epoch%2==0:
                #     cluster_label_rgb_rgb=[]
                #     intersect_count_list=[]

                #     for l in range(part):
                #         ins_sim_rgb_rgb= F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(memory_dy_rgb.features[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #         Score_TOPK = 3#20#10
                #         topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
                #         # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                #         # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
                #         cluster_label_rgb_rgb.append(cluster_indices_rgb_rgb.detach())#.cpu()
                #         if l == 0:
                #             ins_sim_rgb_rgb_all=ins_sim_rgb_rgb
                #         else:
                #             ins_sim_rgb_rgb_all*=ins_sim_rgb_rgb
                #     cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
                #     for n in range(Score_TOPK*part):
                #         intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,n].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                #         intersect_count_list.append(intersect_count)
                #     intersect_count_list = torch.cat(intersect_count_list,1)
                #     intersect_count, _ = intersect_count_list.max(1)
                #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
                #     cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
                #     # print(cluster_label_rgb_rgb)

                #     rgb_rgb_loss = memory_dy_rgb(f_out_all, cluster_label_rgb_rgb)

                # else:
                    cluster_label_rgb_rgb=[]
                    intersect_count_list=[]
                    for l in range(part):
                        ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(memory_dy_ir.features[:,l*768:(l+1)*768].detach().t(), dim=-1))
                        Score_TOPK = 3#20#10
                        topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
                        # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                        # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
                        cluster_label_rgb_rgb.append(cluster_indices_rgb_rgb.detach())#.cpu()
                        if l == 0:
                            ins_sim_rgb_rgb_all=ins_sim_rgb_rgb
                        else:
                            ins_sim_rgb_rgb_all*=ins_sim_rgb_rgb

                    cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
                    for n in range(Score_TOPK*part):
                        intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,n].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
                    # rgb_rgb_loss = memory_dy_ir(f_out_all, cluster_label_rgb_rgb)

                    lamda_cm=0.1
                    update_memory = memory_dy_ir.features[cluster_label_rgb_rgb]

                    self.memory_rgb.features[labels_rgb] = l2norm( lamda_cm*self.memory_rgb.features[labels_rgb] + (1-lamda_cm)*(update_memory) )
                    # self.memory_rgb.features[key[1]] = l2norm( lamda_cm*trainer_interm.memory_rgb.features[key[1]] + (1-lamda_cm)*(update_memory) )





            # confusion_feat_rgb = features_rgb_sim.mm(self.encoder.module.classifier_rgb.weight.data)
            # rgb_rgb_loss = self.criterion_kl(f_out_rgb, Variable(confusion_feat_rgb))
            # rgb_rgb_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))

            # # features_ir =  F.normalize(f_out_ir)#features_ir_.cuda()#
            # features_ir_input = self.encoder.module.classifier_ir(f_out_ir) 
            # features_ir= features_ir_input*20
            # ir_softmax_dim= self.encoder.module.ir_softmax_dim
            # features_ir_1 = F.softmax(features_ir[:,:ir_softmax_dim[0]], dim=1)
            # features_ir_2 = F.softmax(features_ir[:,ir_softmax_dim[0]:], dim=1)
            # features_ir_sim = torch.cat((features_ir_1,features_ir_2), dim=1)
            # ir_ir_loss = self.criterion_ce_soft(features_ir_input,features_ir_sim)
            # # confusion_feat_ir= features_ir_sim.mm(self.encoder.module.classifier_ir.weight.data)
            # # ir_ir_loss = self.criterion_kl(f_out_ir, Variable(confusion_feat_ir))
            # # ir_ir_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))

            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#

            loss_all = self.memory_rgb(f_out_all, labels_all) 
            lamda_i = 0
            loss = loss_all+lamda_c*loss_camera_all#+0.1*rgb_rgb_loss#+ir_ir_loss#+loss_confusion_all#all_all_loss+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+#+ir_ir_loss #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

            # loss = lamda_cc*(loss_ir+loss_rgb)+loss_camera_rgb+loss_camera_ir #+ loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())


            loss_ir_log.update(loss_all.item())
            loss_rgb_log.update(loss_all.item())
            loss_camera_rgb_log.update(loss_camera_all.item())
            loss_camera_ir_log.update(loss_camera_all.item())
            # ir_rgb_loss_log.update(ir_rgb_loss.item())
            # rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            # loss_ins_ir_log.update(loss_ins_ir.item())
            # loss_ins_rgb_log.update(loss_ins_rgb.item())


            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss all {:.3f}\t'
            #           'Loss all {:.3f}\t'
            #           'camera all {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))


            if (i + 1) % print_freq == 0:
                print("ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine")
                print("ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine f_out_rgb Size", f_out_rgb.size())
                print("ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine f_out_ir Size", f_out_ir.size())
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # Note.write('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f} ({:.3f})\t'
            #           'Loss rgb {:.3f} ({:.3f})\t'
            #           'camera ir {:.3f} ({:.3f})\t'
            #           'camera rgb {:.3f} ({:.3f})\t'
            #           'ir_rgb_loss_log {:.3f} ({:.3f})\t'
            #           'rgb_ir_loss_log {:.3f} ({:.3f})\t'
            #           'ir_ir_loss_log {:.3f} ({:.3f})\t'
            #           'rgb_rgb_loss_log {:.3f} ({:.3f})\t\n'
            #           # 'ir_ir_loss_log {:.3f}\t'
            #           # 'rgb_rgb_loss_log {:.3f}\t'
            #           # 'loss_ins_ir_log {:.3f}\t'
            #           # 'loss_ins_rgb_log {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
            #                   loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
            #                   ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
            #                   ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            

                # if epoch >= start_cam:
                # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_tri',loss_tri.item())
                # print('loss_confusion_all',loss_confusion_all.item())
                    # print('loss_intra_ir,loss_inter_ir,intrawise_loss_ir',loss_ins_ir.item(),loss_intra_ir.item(),loss_inter_ir.item(),intrawise_loss_ir.item())
                    # print('loss_intra_rgb,loss_inter_rgb,intrawise_loss_rgb',loss_ins_rgb.item(),loss_intra_rgb.item(),loss_inter_rgb.item(),intrawise_loss_rgb.item())
            # pseudo_labels_all=self.wise_memory_all.labels.numpy()
            # cluster_features_all = self.generate_cluster_features(pseudo_labels_all, self.wise_memory_all.features)

            # num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
            # cam_moment-0.1
            # for cc in torch.unique(cid_all):
            #     # print(cc)
            #     inds = torch.nonzero(cid_all == cc).squeeze(-1)
            #     percam_targets = labels_all[inds]
            #     percam_feat = f_out_all[inds].detach().clone()
 
            #     for k in range(len(percam_feat)):
            #         ori_asso_ind = torch.nonzero(concate_intra_class_rgb == percam_targets[k]).squeeze(-1)
            #         percam_tempV_rgb[ori_asso_ind] = (1-cam_moment)*percam_feat[k]+cam_moment*percam_tempV_rgb[ori_asso_ind]

            # # self.memory_rgb.features = F.normalize(cluster_features_all, dim=1).cuda()
        # Note.close()  
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)

    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV,percam_memory#memory_class_mapper
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper,cross_m=False):
        beta = 0.07#0.07
        bg_knn = 50#100#50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):

            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta

            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                # sel_ind_2 = torch.sort(temp_sims[k])[1][1:bg_knn*10]
                # sel_ind = torch.cat((sel_ind,sel_ind_2), dim=-1)
                # nearest_intra = temp_sims[k].max(dim=-1, keepdim=True)[0]
                # mask_neighbor_intra = torch.gt(temp_sims[k], nearest_intra * 0.8)
                # sel_ind = torch.nonzero(mask_neighbor_intra).squeeze(-1)
                # if cross_m == True:
                #     concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#
                #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                #     torch.device('cuda'))
                #     concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
                # else:
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#target_inputs[k, ori_asso_ind]#
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                torch.device('cuda'))
                # concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
                # print('len(concated_input)',len(concated_input))
                # print('len(ori_asso_ind)',len(ori_asso_ind))
                # concated_target[0:len(concated_input)] = 1.0 / (len(concated_input)+1e-8)
                # concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
                concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam
    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(tgt.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(memory.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        score = F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m)/0.01,dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score,dim=1)
        return score



class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow =torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            
            loss = self.ranking_loss(diff_pow, y)
        
        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss#, correct




class ClusterContrastTrainer_pretrain_joint(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_joint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()




        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)


            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            #_,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            #cid_rgb,cid_ir,index_rgb,index_ir = self._forward(epoch,inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,f_out_rgb_llm,f_out_ir_llm,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,stage2=False)
            


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb# + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print("ClusterContrastTrainer_pretrain_joint")
                print("ClusterContrastTrainer_pretrain_joint f_out_rgb Size:", f_out_rgb.size())
                print("ClusterContrastTrainer_pretrain_joint f_out_ir Size:", f_out_ir.size())
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,stage2=False):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,stage2=stage2)
        #return self.encoder(epoch,x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir,stage2=stage2)
