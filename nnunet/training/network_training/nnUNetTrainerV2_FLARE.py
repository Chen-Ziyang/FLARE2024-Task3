from time import time
import os
import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_UNet_GAN_Seg import Generic_GAN_UNet, Generic_Seg_UNet, PatchDiscriminator
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerUDAV2 import nnUNetTrainerUDAV2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.learning_rate.poly_lr import poly_lr
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch.cuda.amp import autocast


class Style_loss(nn.Module):
    def __init__(self):
        super(Style_loss, self).__init__()

    def gram(self, matrix, norm):
        features = matrix.view(matrix.shape[0], matrix.shape[1], -1)
        features = features / norm
        gram_matrix = torch.matmul(features, features.permute(0, 2, 1).contiguous())
        return gram_matrix

    def cal_loss(self, s1, s2):
        size = np.sqrt(s1.shape[-1] * s1.shape[-2] * s1.shape[-3])
        gram_1 = self.gram(s1, size)
        gram_2 = self.gram(s2, size)
        loss = torch.sum(torch.square(gram_1 - gram_2)) / (4. * s1.shape[1])
        return loss

    def forward(self, Style_1, Style_2):
        if isinstance(Style_1, list) and isinstance(Style_2, list):
            loss = 0
            for i in range(len(Style_1)):
                loss += self.cal_loss(Style_1[i], Style_2[i].detach())
            return loss / len(Style_1)
        else:
            return self.cal_loss(Style_1, Style_2.detach())


class GAN_loss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='wgan', target_real_label=1.0, target_fake_label=0.0):
        super(GAN_loss, self).__init__()
        self.real_label = torch.tensor(target_real_label)
        self.fake_label = torch.tensor(target_fake_label)
        self.gan_mode = gan_mode
        if gan_mode == 'mse':
            self.loss = nn.MSELoss()
        elif gan_mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(prediction.device)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['mse', 'bce']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class nnUNetTrainerV2_FLARE_Big(nnUNetTrainerUDAV2):
    def initialize_network(self):
        self.conv_per_stage = 3
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 1000
        
        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False
        
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = Generic_Seg_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                        self.max_num_features, low_level=3)
        gen_net_num_pool_op_kernel_sizes = [[1,2,2], [2,2,2], [2,2,2], [1,2,2]]
        self.generator = Generic_GAN_UNet(self.num_input_channels, self.base_num_features, 1,
                                          len(gen_net_num_pool_op_kernel_sizes),
                                          2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                          net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                          gen_net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, False,
                                          self.max_num_features, low_level=3)
        self.discriminator = PatchDiscriminator(in_ch=1, base_num_features=self.base_num_features,
                                                pool_op_kernel_sizes=[[1,2,2], [2,2,2], [2,2,2], [1,2,2]],
                                                weightInitializer=InitWeights_He(1e-2))
        if torch.cuda.is_available():
            self.network.cuda()
            self.generator.cuda()
            self.discriminator.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        # Loss Function
        self.recon_loss = torch.nn.MSELoss()
        self.dis_loss = GAN_loss(gan_mode='bce')
        self.content_loss = torch.nn.MSELoss()
        self.style_loss = Style_loss()
        self.seg_consis_loss = torch.nn.MSELoss()

        # Learning Rate
        self.gen_lr = 1e-4
        self.dis_lr = 2.5e-5

    def compute_seg_consis(self, output, target):
        loss = None
        for i in range(len(target)):
            if loss is None:
                loss = self.seg_consis_loss(output[i], target[i].detach())
            else:
                loss += self.seg_consis_loss(output[i], target[i].detach())
        return loss

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

        self.gen_optimizer = torch.optim.SGD(self.generator.parameters(), lr=self.gen_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        self.gen_lr_scheduler = None

        self.dis_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.dis_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        self.dis_lr_scheduler = None

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_elastic"] = True

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.gen_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.gen_lr, 0.9)
        self.dis_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.dis_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        self.print_to_log_file("gen_lr:", np.round(self.gen_optimizer.param_groups[0]['lr'], decimals=6))
        self.print_to_log_file("dis_lr:", np.round(self.dis_optimizer.param_groups[0]['lr'], decimals=6))


class nnUNetTrainerV2_FLARE_Pseudo(nnUNetTrainerV2):
    def initialize_network(self):
        self.conv_per_stage = 3
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 1000

        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["do_elastic"] = True


class nnUNetTrainerV2_FLARE_Small(nnUNetTrainerV2):
    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 1000  # Select the model of the 200th epoch for testing
        self.initial_lr = 1e-2
        
        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num-1]

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, None,
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["do_elastic"] = True
