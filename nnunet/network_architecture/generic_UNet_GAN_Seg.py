#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.nd_sigmoid import sigmoid_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import NeuralNetwork
import torch.nn.functional
from nnunet.network_architecture.generic_UNet import Generic_UNet


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_Seg_UNet(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, low_level=3):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_Seg_UNet, self).__init__(input_channels, base_num_features, num_classes, num_pool,
                                               num_conv_per_stage, feat_map_mul_on_downscale, conv_op, norm_op,
                                               norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                               deep_supervision, dropout_in_localization, final_nonlin,
                                               weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling,
                                               max_num_features, basic_block, seg_output_use_bias)
        self.low_level = low_level
        self.recon_tu, self.conv_blocks_recon = [], []

        convolutional_upsampling = False
        recon_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        for u in range(num_pool):
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            if u != num_pool - 1 and not convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.recon_tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode='trilinear'))

            recon_conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            recon_conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_recon.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, recon_conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, recon_conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        self.recon_tu = nn.ModuleList(self.recon_tu)
        self.conv_blocks_recon = nn.ModuleList(self.conv_blocks_recon)
        self.recon_head = conv_op(self.conv_blocks_recon[-1][-1].output_channels, 1,
                                  1, 1, 0, 1, 1, seg_output_use_bias)

    def forward(self, x, tar_x=None, tar_seg=False, shallow_feat=False, last_feat=False):
        if tar_x is not None:
            func_outputs = []
            tar_skips = []
            tar_x = -1. + 2. * (tar_x - tar_x.min()) / (tar_x.max() - tar_x.min() + 1e-10)
            # Encoder
            for d in range(len(self.conv_blocks_context) - 1):
                tar_x = self.conv_blocks_context[d](tar_x)
                tar_skips.append(tar_x)
                if not self.convolutional_pooling:
                    tar_x = self.td[d](tar_x)

            tar_x = self.conv_blocks_context[-1](tar_x)

            # Recon Decoder
            recon_x = self.recon_tu[0](tar_x)
            recon_x = torch.cat((recon_x, tar_skips[-(0 + 1)]), dim=1)
            recon_x = self.conv_blocks_recon[0](recon_x)
            for u in range(1, len(self.recon_tu)):
                recon_x = self.recon_tu[u](recon_x)
                recon_x = torch.cat((recon_x, tar_skips[-(u + 1)]), dim=1)
                recon_x = self.conv_blocks_recon[u](recon_x)
            recon_tar_x = self.recon_head(recon_x)

            func_outputs.append(tar_skips[:self.low_level])
            func_outputs.append(recon_tar_x)
            if tar_seg:
                seg_outputs = []
                # Seg Decoder
                for u in range(len(self.tu)):
                    tar_x = self.tu[u](tar_x)
                    tar_x = torch.cat((tar_x, tar_skips[-(u + 1)]), dim=1)
                    tar_x = self.conv_blocks_localization[u](tar_x)
                    seg_outputs.append(self.final_nonlin(self.seg_outputs[u](tar_x)))

                # Deep supervision or not
                if self._deep_supervision and self.do_ds:
                    func_outputs.append(tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1],
                                                                                        seg_outputs[:-1][::-1])]))
                else:
                    func_outputs.append(seg_outputs[-1])
            return func_outputs

        # Normalize to [-1, 1]
        x = -1. + 2. * (x - x.min()) / (x.max() - x.min() + 1e-10)
        skips = []
        seg_outputs = []

        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        last_feature = x

        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        func_outputs = []
        # Deep supervision or not
        if self._deep_supervision and self.do_ds:
            func_outputs.append(tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1],
                                       seg_outputs[:-1][::-1])]))
        else:
            func_outputs.append(seg_outputs[-1])
        if shallow_feat:
            func_outputs.append(skips[:self.low_level])
        if last_feat:
            func_outputs.append(last_feature)

        if len(func_outputs) == 1:
            return func_outputs[0]
        else:
            return func_outputs


class Generic_GAN_UNet(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=sigmoid_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, low_level=3):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_GAN_UNet, self).__init__(input_channels, base_num_features, num_classes, num_pool,
                                               num_conv_per_stage, feat_map_mul_on_downscale, conv_op, norm_op,
                                               norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                               deep_supervision, dropout_in_localization, final_nonlin,
                                               weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling,
                                               max_num_features, basic_block, seg_output_use_bias)
        self.seg_outputs = self.seg_outputs[-1]
        self.low_level = low_level

        output_features = base_num_features
        for d in range(low_level):
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)
        input_features = output_features

        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[low_level - 1 - 1]
        else:
            first_stride = None
        self.conv_blocks_context[low_level] = StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                                self.conv_op, self.conv_kwargs, self.norm_op,
                                                                self.norm_op_kwargs, self.dropout_op,
                                                                self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                                first_stride, basic_block=basic_block)

    def forward(self, x, sty_fea):
        # Normalize to [-1, 1]
        x = -1. + 2. * (x - x.min()) / (x.max() - x.min() + 1e-10)
        skips = []

        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if (d + 1) == self.low_level: x = torch.cat((x, sty_fea), dim=1)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)

        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        gen_output = self.final_nonlin(self.seg_outputs(x))
        return gen_output


class PatchDiscriminator(NeuralNetwork):
    def __init__(self, in_ch, base_num_features, pool_op_kernel_sizes, weightInitializer):
        super(PatchDiscriminator, self).__init__()
        filter_num_list = [in_ch, base_num_features, base_num_features*2, base_num_features*4, base_num_features*8, 1]
        self.conv1 = nn.Sequential(
            nn.Conv3d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=pool_op_kernel_sizes[0], padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=pool_op_kernel_sizes[1], padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=pool_op_kernel_sizes[2], padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=pool_op_kernel_sizes[3], padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_final = nn.Conv3d(filter_num_list[-2], filter_num_list[-1], kernel_size=4, stride=1, padding=2, bias=True)
        if weightInitializer is not None:
            self.apply(weightInitializer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_final(x)
        return x
