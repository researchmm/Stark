from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
import torch.nn as nn
import torch.nn.functional as F


class STARKLightningXActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        if self.settings.distill_loss_type == "KL":
            self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        elif self.settings.distill_loss_type == "MSE":
            self.distill_loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported distillation loss type")

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
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (batch, 4) (x1,y1,w,h)

        # compute losses
        if self.settings.deep_sup:
            loss_list, status = [], None
            for pred_boxes in out_dict["pred_boxes"]:
                loss, status = self.compute_losses(pred_boxes, gt_bboxes[0])
                loss_list.append(loss)
            loss = torch.sum(torch.stack(loss_list))
            if self.settings.distill:
                distill_loss = self.compute_distill_losses(out_dict)
                status["loss/distill"] = distill_loss.item()
                loss = loss + distill_loss
            return loss, status
        else:
            pred_boxes = out_dict["pred_boxes"]
            loss, status = self.compute_losses(pred_boxes, gt_bboxes[0])
            return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=template_img_i, mask=template_att_i,
                                           mode='backbone', zx="template%d" % i))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.net(img=search_img, mask=search_att, mode='backbone', zx="search"))

        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list)
        out_dict, _, _ = self.net(seq_dict=seq_dict, mode="transformer", run_box_head=run_box_head,
                                  run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_boxes, gt_bbox, return_status=True):
        # Get boxes
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes)  # (B,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)  # (B,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (B,4) (B,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_distill_losses(self, output_dict):
        ptl_list, pbr_list = output_dict["prob_tl"], output_dict["prob_br"]
        ptl_label, pbr_label = ptl_list[-1], pbr_list[-1]
        n_layer = len(ptl_list) - 1
        dis_loss_list = []
        for i in range(n_layer):
            if self.settings.distill_loss_type == "KL":
                dis_loss_tl = self.distill_loss(F.log_softmax(ptl_list[i], dim=1), ptl_label)
                dis_loss_br = self.distill_loss(F.log_softmax(pbr_list[i], dim=1), pbr_label)
            elif self.settings.distill_loss_type == "MSE":
                dis_loss_tl = self.distill_loss(F.softmax(ptl_list[i], dim=1), ptl_label)
                dis_loss_br = self.distill_loss(F.softmax(pbr_list[i], dim=1), pbr_label)
            else:
                raise ValueError("Unsupported distillation loss type")
            dis_loss = (dis_loss_tl + dis_loss_br) / 2
            dis_loss_list.append(dis_loss)
        return torch.mean(torch.stack(dis_loss_list))
