from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from util_mm.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead

###################################
# Adversarial Autoencoder, AAE
###################################
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, in_channels):
        super(Decoder, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z_out = self.decoder(z)
        return z_out

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.discriminator(z)

class AAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dim):
        super(AAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels)
        self.discriminator = Discriminator(latent_dim, hidden_dim)

    def add_noise(self, input: Tensor, noise_factor: float = 0.1) -> Tensor: 
        noise = torch.randn_like(input) * noise_factor
        noisy_input = input + noise
        return torch.clamp(noisy_input, 0., 1.)

    def add_mask(self, input: torch.Tensor, mask_prob: float = 0.5) -> torch.Tensor: 
        mask = torch.rand_like(input)
        bool_mask = mask > mask_prob
        float_mask = bool_mask.float()
        masked_input = input * float_mask
        return masked_input

    def forward(self, input):
        input = self.add_noise(input)  # NAAE
        # input = self.add_mask(input) # MAAE
        z = self.encoder(input)
        rec = self.decoder(z)
        return [rec, z, input]
    
    def loss_function(self, input, rec, z):
        recons_loss = F.mse_loss(rec, input)
        # print('recons_loss:', recons_loss)
        real_labels = torch.ones(z.size(0), 1).to(z.device)
        fake_labels = torch.zeros(z.size(0), 1).to(z.device)
        real_z = torch.randn(z.size()).to(z.device)
        d_real = self.discriminator(real_z)
        d_fake = self.discriminator(z)
        d_loss = F.binary_cross_entropy(d_real, real_labels) + F.binary_cross_entropy(d_fake, fake_labels)
        g_loss = F.binary_cross_entropy(d_fake, real_labels)
        loss = recons_loss + 0.5*d_loss + 0.05*g_loss 

        # return {'loss_aae': loss, 'd_loss': d_loss, 'g_loss': g_loss}
        return {'loss_aae': loss}


@HEADS.register_module()
class AMLRoIHead(MetaRCNNRoIHead):    
    def __init__(self, inp_dim=2048, latent_dim=2048, hidden_dim=2048, *args, **kargs) -> None:
        
        super().__init__(*args, **kargs)
        self.aae = AAE(inp_dim, latent_dim, hidden_dim)
    
    def _bbox_forward(self, 
                      query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.
        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).
        Returns: dict: A dictionary of predicted results.
        """
        
        # feature aggregation
        roi_feats = self.aggregation_layer(query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
                                           support_feat=support_roi_feats.view(1, -1, 1, 1))[0]

        cls_score, bbox_pred = self.bbox_head(roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)

        return bbox_results

    def _bbox_forward_train(self, query_feats: List[Tensor], 
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.
        Args:
            query_feats (list[Tensor]): List of query features, each item with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where each dict has: 'img_shape', 'scale_factor', 'flip', and may also contain 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'. 
                For details on the values of these keys see `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to each box of support images.
        Returns: dict: Predicted results and losses.
        """
    
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        
        support_feat_rec, support_feat_inv, _ = self.aae(support_feat)

        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge

            random_index = np.random.choice(range(len(support_gt_labels)))
            random_query_label = support_gt_labels[random_index]

            for i in range(support_feat.size(0)):  # 15
                if support_gt_labels[i] == random_query_label:

                    bbox_results = self._bbox_forward(query_roi_feats[start:end],                
                                                      support_feat_inv[i].sigmoid().unsqueeze(0))

                    single_loss_bbox = self.bbox_head.loss(
                                                    bbox_results['cls_score'], bbox_results['bbox_pred'],
                                                    query_rois[start:end], labels[start:end],
                                                    label_weights[start:end], bbox_targets[start:end],
                                                    bbox_weights[start:end])
  
                    for key in single_loss_bbox.keys(): 
                                                    loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc': 
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else: loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_rec)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                                            meta_cls_score, meta_cls_labels,
                                            torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        loss_vae = self.aae.loss_function(support_feat, support_feat_rec, support_feat_inv)
        loss_bbox.update(loss_vae)

        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.
        Args:
            query_feats (list[Tensor]): Features of query image, each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features used for inference only, each key is the class id and value is the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each dict has: `img_shape`, `scale_factor`, `flip`, and may also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space. Default: False.
        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list with shape (num_boxes, 4) and with shape (num_boxes, ) in second list. 
            The length of both lists should be equal to batch_size.
        """
        
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            
            support_feat_rec, support_feat_inv, _ = self.aae(support_feat)

            bbox_results = self._bbox_forward(query_roi_feats, support_feat_inv.sigmoid())
            cls_scores_dict[class_id] =  bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] =  bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
