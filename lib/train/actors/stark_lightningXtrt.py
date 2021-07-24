from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import get_qkv
import torch.nn as nn
import torch.nn.functional as F


class STARKLightningXtrtActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

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
        q, k, v, key_padding_mask = get_qkv(feat_dict_list)
        out_dict, _, _ = self.net(q=q, k=k, v=v, key_padding_mask=key_padding_mask, mode="transformer")
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
