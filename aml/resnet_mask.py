# Copyright (c) HIT. All rights reserved.
from typing import Tuple
from mmcv.cnn import build_conv_layer
from mmdet.models.builder import BACKBONES
from torch import Tensor
from mmdet.models import ResNet

@BACKBONES.register_module()
class MaskResNet(ResNet):
    """ResNet with `meta_conv` to handle different inputs in metarcnn and fsdetview.
    When input with shape (N, 3, H, W) from images, the network will use `conv1` as regular ResNet. 
    When input with shape (N, 4, H, W) from (image + mask) the network will replace `conv1` with `sup_conv` to handle additional channel.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sup_conv = build_conv_layer(
            self.conv_cfg,  # from config of ResNet
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)

    def forward(self, x: Tensor, roi_mask: bool = False) -> Tuple[Tensor]:
        """Forward function.
        When input with shape (N, 3, H, W) from images, the network will use `conv1` as regular ResNet. 
        When input with shape (N, 4, H, W) from (image + mask) the network will replace `conv1` with `sup_conv` to handle additional channel.
        Args:
            x (Tensor): Tensor with shape (N, 3, H, W) from images or (N, 4, H, W) from (images + masks).
            roi_mask (bool): If set True, forward input tensor with `ROI_Mask` which require tensor with shape (N, 3, H, W).
        Returns:
            tuple[Tensor]: Tuple of features, each item with shape (N, C, H, W).
        """
        if roi_mask:
            fourth_channel = x[:, 3, :, :]
            mask = fourth_channel > 0
            mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
            first_three_channels = x[:, :3, :, :]
            x = first_three_channels * mask.float()
            x = self.sup_conv(x)
        else:
            x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
