# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from MinkowskiEngine import MinkowskiReLU, MinkowskiGlobalPooling
from MinkowskiEngine import SparseTensor

import torch
import torch.nn as nn
from torch.serialization import default_restore_location

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork

from model.common import ConvType, NormType, get_norm, conv, sum_pool
from model.resnet_block import BasicBlock, Bottleneck
from lib.utils import checkpoint, precision_at_one, Timer, AverageMeter, get_prediction, load_state_with_same_shape, count_parameters

import logging


class ClusterLabelModel(MinkowskiNetwork):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
    HAS_LAST_BLOCK = False

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0
        print("D:")
        print(D)
        MinkowskiNetwork.__init__(self, D)
        # super(MinkowskiNetwork, self).__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config

        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()
        self.normalize_feature = config.net.normalize_feature

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            # Should we also initialize the Minkowski Convolution?

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiReLU() # Switched to ReLU instead of LeakyReLU because v0.4.3 doesn't have leaky
            # ME.MinkowskiLeakyReLU(), # Should this be relu to align with the rest of the network?
        )

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_type=NormType.BATCH_NORM,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D))

        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, config, D):
        dilations = self.DILATIONS
        bn_momentum = config.opt.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.net.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(self.LAYERS[3] * 2, config.net.mlp_dim),
            ME.MinkowskiDropout(),
            self.get_mlp_block(config.net.mlp_dim, config.net.mlp_dim),
            ME.MinkowskiLinear(config.net.mlp_dim, out_channels, bias=True)
        )

        self.relu = MinkowskiReLU(inplace=True)
        # self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        encoder_out = self.block4(out)

        x1 = self.global_max_pool(encoder_out)
        x2 = self.global_avg_pool(encoder_out)

        contrastive = self.final(ME.cat(x1, x2))
        if self.normalize_feature:
            contrastive = SparseTensor(
                contrastive.F / torch.norm(contrastive.F, p=2, dim=1, keepdim=True),
                coords_key=contrastive.coords_key,
                coords_manager=contrastive.coords_man)

        return contrastive

    def updateWithPretrainedWeights(self, weights_file):
        # Load weights if specified by the parameter.

        print("Before")
        print(self.state_dict())
        if weights_file != '':
            logging.info('===> Loading weights: ' + weights_file)
            state = torch.load(weights_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            matched_weights = load_state_with_same_shape(self, state['state_dict'])
            model_dict = self.state_dict()
            model_dict.update(matched_weights)
            self.load_state_dict(model_dict)
            print("After")
            print(self.state_dict())
        else:
            logging.info('Weights file name was empty')
