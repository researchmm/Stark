from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import get_qkv, merge_template_search
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss


class STARKLightningXtrtdistillActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""

    def __init__(self, net, objective, loss_weight, settings, net_teacher):
        super().__init__(net, objective)
        self.net_teacher = net_teacher
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

        if "KL" in self.settings.distill_loss_type:
            print("Distill model with KL Loss")
            self.distill_loss_kl = nn.KLDivLoss(reduction="batchmean")
        if "L1" in self.settings.distill_loss_type:
            print("Distill model with L1 Loss")

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward student
        out_dict = self.forward_pass(self.net, data)
        # forward teacher
        out_dict_teacher = self.forward_pass_teacher(self.net_teacher, data, True, True)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_teacher, gt_bboxes[0])
        return loss, status

    def forward_pass(self, net, data):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(net(img=template_img_i, mask=template_att_i,
                                      mode='backbone', zx="template%d" % i))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(net(img=search_img, mask=search_att, mode='backbone', zx="search"))

        # run the transformer and compute losses
        q, k, v, key_padding_mask = get_qkv(feat_dict_list)
        # for student network, here we output the original logits without softmax
        out_dict, _, _ = net(q=q, k=k, v=v, key_padding_mask=key_padding_mask, mode="transformer", softmax=False)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def forward_pass_teacher(self, net, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(net(img=template_img_i, mask=template_att_i,
                                      mode='backbone', zx="template%d" % i))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(net(img=search_img, mask=search_att, mode='backbone', zx="search"))

        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list)
        out_dict, _, _ = net(seq_dict=seq_dict, mode="transformer", run_box_head=run_box_head,
                             run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, out_dict, out_dict_teacher, gt_bbox, return_status=True):
        pred_boxes = out_dict["pred_boxes"]
        pred_boxes_teacher = out_dict_teacher["pred_boxes"]
        # Get boxes
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes)  # (B,4) (x1,y1,x2,y2)
        pred_boxes_vec_teacher = box_cxcywh_to_xyxy(pred_boxes_teacher)  # (B,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)  # (B,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (B,4) (B,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        try:
            _, iou_teacher = self.objective['giou'](pred_boxes_vec_teacher, gt_boxes_vec)
        except:
            iou_teacher = torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute distillation loss
        dis_loss_l1, dis_loss_kl = self.compute_distill_losses(out_dict, out_dict_teacher)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['l1'] * dis_loss_l1 + self.loss_weight['l1'] * dis_loss_kl
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            mean_iou_teacher = iou_teacher.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/distill_l1": dis_loss_l1.item(),
                      "Loss/distill_kl": dis_loss_kl.item(),
                      "IoU": mean_iou.item(),
                      "IoU_teacher": mean_iou_teacher.item()}
            return loss, status
        else:
            return loss

    def compute_distill_losses(self, out_dict, out_dict_t):
        ptl, pbr = out_dict["prob_tl"], out_dict["prob_br"]
        ptl_t, pbr_t = out_dict_t["prob_tl"], out_dict_t["prob_br"]
        dis_loss_l1, dis_loss_kl = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        if "KL" in self.settings.distill_loss_type:
            dis_loss_kl_tl = self.distill_loss_kl(F.log_softmax(ptl, dim=1), ptl_t)
            dis_loss_kl_br = self.distill_loss_kl(F.log_softmax(pbr, dim=1), pbr_t)
            dis_loss_kl = (dis_loss_kl_tl + dis_loss_kl_br) / 2
        if "L1" in self.settings.distill_loss_type:
            dis_loss_l1_tl = l1_loss(F.softmax(ptl), ptl_t, reduction="sum") / self.bs
            dis_loss_l1_br = l1_loss(F.softmax(pbr), pbr_t, reduction="sum") / self.bs
            dis_loss_l1 = (dis_loss_l1_tl + dis_loss_l1_br) / 2
        return dis_loss_l1, dis_loss_kl
