"""
2021.06.24 STARK Lightning-X Model (Spatial-only).
2021.06.27 for converting pytorch model to trt model
"""
import torch
from torch import nn
import torch.nn.functional as F
from .backbone_X import build_backbone_x
from .position_encoding import build_position_encoding_new
from .lite_encoder import build_lite_encoder  # encoder only
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import time


class STARKLightningXtrt(nn.Module):
    """Modified from stark_s_plus_sp
    The goal is to achieve ultra-high speed (1000FPS)
    2021.06.24 We change the input datatype to standard Tensor, rather than NestedTensor
    2021.06.27 Definition of transformer is changed"""
    def __init__(self, backbone, transformer, box_head, pos_emb_z, pos_emb_x, head_type="CORNER_LITE",
                 distill=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.pos_emb_x = pos_emb_x
        for i in range(len(pos_emb_z)):
            setattr(self, "pos_emb_z%d" % i, pos_emb_z[i])
        hidden_dim = transformer.d_model
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.head_type = head_type
        self.distill = distill
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img=None, mask=None, q=None, k=None, v=None, key_padding_mask=None,
                mode="backbone", zx="template", softmax=True):
        if mode == "backbone":
            return self.forward_backbone(img, zx, mask)
        elif mode == "transformer":
            return self.forward_transformer(q, k, v, key_padding_mask=key_padding_mask, softmax=softmax)
        else:
            raise ValueError

    def forward_backbone(self, img: torch.Tensor, zx: str, mask: torch.Tensor):
        """The input type is standard tensor
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(img, torch.Tensor)
        """run the backbone"""
        output_back = self.backbone(img)  # features & masks, position embedding for the search
        """get the positional encoding"""
        bs = img.size(0)  # batch size
        if zx == "search":
            pos = self.pos_emb_x(bs)
        elif "template" in zx:
            pos_emb_z = getattr(self, "pos_emb_z%d" % int(zx[-1]))
            pos = pos_emb_z(bs)
        else:
            raise ValueError("zx should be 'template_0' or 'search'.")
        """get the downsampled attention mask"""
        mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
        """adjust the shape"""
        return self.adjust(output_back, pos, mask_down)

    def forward_transformer(self, q, k, v, key_padding_mask=None, softmax=True):
        # run the transformer encoder
        enc_mem = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
        # run the corner head
        if self.distill:
            outputs_coord, prob_tl, prob_br = self.forward_box_head(enc_mem, softmax=softmax)
            return {"pred_boxes": outputs_coord, "prob_tl": prob_tl, "prob_br": prob_br}, None, None
        else:
            # s = time.time()
            out, outputs_coord = self.forward_box_head(enc_mem)
            # e = time.time()
            # print("head time: %.1f ms" % ((e-s)*1000))
            return out, outputs_coord, None

    def forward_box_head(self, memory, softmax=True):
        """ memory: encoder embeddings (HW1+HW2, B, C) / (HW2, B, C)"""
        if "CORNER" in self.head_type:
            # encoder output for the search region (H_x*W_x, B, C)
            fx = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
            fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
            # run the corner head
            if self.distill:
                coord_xyxy, prob_vec_tl, prob_vec_br = self.box_head(fx_t, return_dist=True, softmax=softmax)
                outputs_coord = box_xyxy_to_cxcywh(coord_xyxy)
                return outputs_coord, prob_vec_tl, prob_vec_br
            else:
                outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
                out = {'pred_boxes': outputs_coord}
                return out, outputs_coord
        else:
            raise ValueError

    def adjust(self, src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor):
        """
        """
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}


def build_stark_lightning_x_trt(cfg, phase: str):
    """phase: 'train' or 'test'
    during the training phase, we need to
        (1) load backbone pretrained weights
        (2) freeze some layers' parameters"""
    backbone = build_backbone_x(cfg, phase=phase)
    transformer = build_lite_encoder(cfg)
    box_head = build_box_head(cfg)
    fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
    pos_emb_x = build_position_encoding_new(cfg, fsz_x)
    pos_emb_z = build_position_encoding_new(cfg, fsz_z)
    model = STARKLightningXtrt(
        backbone,
        transformer,
        box_head,
        [pos_emb_z],
        pos_emb_x,
        head_type=cfg.MODEL.HEAD_TYPE,
        distill=cfg.TRAIN.DISTILL
    )

    return model


class STARKLightningXtrt_new(STARKLightningXtrt):
    def __init__(self, backbone, transformer, box_head, pos_emb_z, pos_emb_x, head_type="CORNER_LITE",
                 distill=False):
        super(STARKLightningXtrt_new, self).__init__(backbone, transformer, box_head, pos_emb_z, pos_emb_x,
                                                     head_type=head_type, distill=distill)

    def forward_backbone(self, img: torch.Tensor, zx: str, mask: torch.Tensor, backbone):
        """The input type is standard tensor
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(img, torch.Tensor)
        """run the backbone"""
        output_back = backbone(img)  # features & masks, position embedding for the search
        """get the positional encoding"""
        bs = img.size(0)  # batch size
        if zx == "search":
            pos = self.pos_emb_x(bs)
        elif "template" in zx:
            pos_emb_z = getattr(self, "pos_emb_z%d" % int(zx[-1]))
            pos = pos_emb_z(bs)
        else:
            raise ValueError("zx should be 'template_0' or 'search'.")
        """get the downsampled attention mask"""
        mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
        """adjust the shape"""
        return self.adjust(output_back, pos, mask_down)


def build_stark_lightning_x_trt_new(cfg, phase: str):
    """phase: 'train' or 'test'
    during the training phase, we need to
        (1) load backbone pretrained weights
        (2) freeze some layers' parameters"""
    backbone = build_backbone_x(cfg, phase=phase)
    transformer = build_lite_encoder(cfg)
    box_head = build_box_head(cfg)
    fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
    pos_emb_x = build_position_encoding_new(cfg, fsz_x)
    pos_emb_z = build_position_encoding_new(cfg, fsz_z)
    model = STARKLightningXtrt_new(
        backbone,
        transformer,
        box_head,
        [pos_emb_z],
        pos_emb_x,
        head_type=cfg.MODEL.HEAD_TYPE,
        distill=cfg.TRAIN.DISTILL
    )

    return model