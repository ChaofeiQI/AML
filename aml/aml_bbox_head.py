import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS
from util_mm.detection.models.roi_heads.bbox_heads.meta_bbox_head import MetaBBoxHead
from mmdet.models.roi_heads import BBoxHead

@HEADS.register_module()
class AMLBBoxHead(MetaBBoxHead, BBoxHead):
    @auto_fp16()
    def forward(self, x_agg, x_query):
        '''
        x_agg: torch.Size([128, 2048])
        support_feat: torch.Size([128, 2048])
        '''
        if self.with_avg_pool:
            if x_agg.numel() > 0:
                x_agg = self.avg_pool(x_agg)
                x_agg = x_agg.view(x_agg.size(0), -1)
            else:
                # avg_pool does not support empty tensor, so use torch.mean instead it
                x_agg = torch.mean(x_agg, dim=(-1, -2))
            if x_query.numel() > 0:
                x_query = self.avg_pool(x_query)
                x_query = x_query.view(x_query.size(0), -1)
            else:
                x_query = torch.mean(x_query, dim=(-1, -2))
        cls_score = self.fc_cls(x_agg) if self.with_cls else None
        bbox_pred = self.fc_reg(x_query) if self.with_reg else None
        # print('cls_scores:{}, bbox_pred:{}'.format(cls_score.shape, bbox_pred.shape))  # cls_scores:torch.Size([128, 21]), bbox_pred:torch.Size([128, 80])
        return cls_score, bbox_pred
