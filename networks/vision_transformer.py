# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
from .cswin_msa import CSWinTransformer

# from dynamic_tanh import DynamicTanh

logger = logging.getLogger(__name__)


class CSwinMSA(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(CSwinMSA, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.cswin_msa = CSWinTransformer(img_size=config.DATA.IMG_SIZE,
                                           patch_size=config.MODEL.CSWIN.PATCH_SIZE,
                                           in_chans=config.MODEL.CSWIN.IN_CHANS,
                                           num_classes=self.num_classes,
                                           embed_dim=config.MODEL.CSWIN.EMBED_DIM,
                                           depth=config.MODEL.CSWIN.DEPTH,
                                           split_size=config.MODEL.CSWIN.SPLIT_SIZE,
                                           num_heads=config.MODEL.CSWIN.NUM_HEADS,
                                           mlp_ratio=config.MODEL.CSWIN.MLP_RATIO,
                                           qkv_bias=config.MODEL.CSWIN.QKV_BIAS,
                                           qk_scale=config.MODEL.CSWIN.QK_SCALE,
                                           drop_rate=config.MODEL.DROP_RATE,
                                           drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                           # norm_layer=DynamicTanh
                                           )
        torch.save(self.cswin_msa.state_dict(), 'cswin_msa.pth')
        print("CSWinMSA model is saved.")

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)

        logits = self.cswin_msa(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        # print('pretrained_path')
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            pretrained_dict = pretrained_dict['state_dict_ema']
            model_dict = self.cswin_msa.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "stage" in k:
                    current_k = "stage_up" + k[5:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k].shape))
                        del full_dict[k]

            msg = self.cswin_msa.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
