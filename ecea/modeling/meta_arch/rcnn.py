import torch
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from ecea.modeling.roi_heads import build_roi_heads
from detectron2.layers import Conv2d, ShapeSpec, get_norm

#import  from MMDetection start
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_positional_encoding)
from mmdet.models.utils.builder import TRANSFORMER

from torch.nn.init import normal_
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
#import from MMDetection end


__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        

        if cfg.MODEL.BACKBONE.WITHECEA:
            # add transofomer encoder
            
            in_channels_trans = 2048
            num_query=300
            num_feature_levels=4

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True)
            in_channels = 1024
            out_channels = 256
            
            
            norm = ""
            use_bias = norm == ""
            self.lateral_conv_in = Conv2d( in_channels, out_channels, kernel_size=1, bias=use_bias )
            self.lateral_conv_out = Conv2d( out_channels, in_channels, kernel_size=1, bias=use_bias ) 
            
            weight_init.c2_xavier_fill(self.lateral_conv_in)
            weight_init.c2_xavier_fill(self.lateral_conv_out)
            
            encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=4,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256, num_levels = len(self._SHAPE_) ),#num_levels必须等于num_outs
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')))
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
            
            num_outs = len(self._SHAPE_) 
            self.num_feature_levels = out_channels
            
            self.encoder = build_transformer_layer_sequence(encoder)
            self.embed_dims = self.encoder.embed_dims
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.positional_encoding = build_positional_encoding( positional_encoding )
            
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        self.to(self.device) 
            
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")
        if cfg.MODEL.BACKBONE.FREEZE_ECEA:
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            # for p in self.reference_points.requires_grad_:
            #     p.requires_grad = False
                
            # for p in self.lateral_conv_in:
            #     p.requires_grad = False
                
            # for p in self.lateral_conv_out:
            #     p.requires_grad = False
                
            # for p in self.level_embeds:
            #     p.requires_grad = False
                
            # for p in self.positional_encoding:
            #     p.requires_grad = False
                
            print("froze PAM parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)    
          
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.cfg.MODEL.BACKBONE.WITHECEA:
            results = [self.lateral_conv_in(features[key]) for key in features.keys()]
            
            #apply trans encoder
            mlvl_feats = tuple(results)
            batch_input_shape = tuple(mlvl_feats[0].shape[-2:])
            input_img_h, input_img_w = batch_input_shape
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            
            for img_id in range(batch_size):
                img_h, img_w = mlvl_feats[0].shape[-2:]
                img_masks[img_id, :img_h, :img_w] = 0
            
            mlvl_masks = []
            mlvl_positional_encodings = []
            
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))
            query_embeds = None
            for out in results:
                if not torch.isfinite(out).all():
                    a = 1
            feat_flatten = [out.flatten(2) for out in results]     # outs: list( [BN, channel, width, height] )
                                                                # feat_flatten [BN, channel, sum(width*height)]
            spatial_shapes = [out.shape[2:4] for out in results]
            
            mask_flatten = [mask.flatten(1) for mask in mlvl_masks]
            lvl_pos_embed_flatten = []
            
            for lvl, ( mask, pos_embed) in enumerate(
                    zip( mlvl_masks, mlvl_positional_encodings)):

                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

                
            feat_flatten = torch.cat(feat_flatten, 2)
            if torch.isnan(feat_flatten).any():
                a = 1
            if torch.isinf(feat_flatten).any():
                a = 1
            mask_flatten = torch.cat(mask_flatten, 1)
            
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor( spatial_shapes, dtype=torch.long, device=lvl_pos_embed_flatten.device )
            
            level_start_index = torch.cat((spatial_shapes.new_zeros( (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            valid_ratios = torch.stack( [ self.get_valid_ratio(m) for m in mlvl_masks], 1)
            
            reference_points = self.get_reference_points(spatial_shapes,valid_ratios, device=feat.device)
                
            feat_flatten = feat_flatten.permute(2, 0, 1)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            kwargs = {}
            memory = self.encoder(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                **kwargs)
            feat_flatten = memory.permute(1, 2, 0)
            
            bn,channel,_ = feat_flatten.shape
            value_list = feat_flatten.split([H_ * W_ for H_, W_ in spatial_shapes],
                                dim=2)

            out_results = [torch.reshape(value_list[i],[bn, channel, spatial_shapes[i][0], spatial_shapes[i][1]] ) for i in range(len(value_list))]
            
            for f, res in zip(features, out_results):
                features[f] = features[f] * 0.6 + self.lateral_conv_out(res) * 0.4

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        if spatial_shapes.shape[0] == 1:
            lvl = 0
            H, W = spatial_shapes[0]
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        else:
            
            for lvl, (H, W) in enumerate(spatial_shapes):
                #  TODO  check this 0.5
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=torch.float32, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=torch.float32, device=device))
                ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
                ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos
