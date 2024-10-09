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
import torch
import torch.nn as nn
import numpy as np
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_UNet_GAN_Seg import Generic_GAN_UNet, Generic_Seg_UNet, PatchDiscriminator
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.network_training.nnUNetTrainerV2_DDP_UDA import nnUNetTrainerV2_DDP_UDA
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.training.network_training.nnUNetTrainerV2_FLARE import GAN_loss
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from nnunet.utilities.nd_softmax import softmax_helper
from torch.cuda.amp import autocast
from fvcore.nn import FlopCountAnalysis, flop_count_table


class nnUNetTrainerV2_FLARE_Small_DDP(nnUNetTrainerV2_DDP):
    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 100
        self.save_every = self.max_num_epochs // 10

        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]

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
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, None,
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["do_elastic"] = True


# class nnUNetTrainerV2_FLARE_Pseudo_DDP(nnUNetTrainerV2_DDP_UDA):
#     def initialize_network(self):
#         self.conv_per_stage = 3
#         self.base_num_features = 64
#         self.max_num_features = 1024
#         self.max_num_epochs = 1000
#
#         # 取消验证，加速训练
#         self.num_val_batches_per_epoch = 1
#         self.save_best_checkpoint = False
#
#         if self.threeD:
#             conv_op = nn.Conv3d
#             dropout_op = nn.Dropout3d
#             norm_op = nn.InstanceNorm3d
#         else:
#             conv_op = nn.Conv2d
#             dropout_op = nn.Dropout2d
#             norm_op = nn.InstanceNorm2d
#
#         norm_op_kwargs = {'eps': 1e-5, 'affine': True}
#         dropout_op_kwargs = {'p': 0, 'inplace': True}
#         net_nonlin = nn.LeakyReLU
#         net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
#         self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
#                                     len(self.net_num_pool_op_kernel_sizes),
#                                     self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
#                                     dropout_op_kwargs,
#                                     net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
#                                     self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
#                                     self.max_num_features)
#         if torch.cuda.is_available():
#             self.network.cuda()
#         self.network.inference_apply_nonlin = softmax_helper
#
#     def initialize_optimizer_and_scheduler(self):
#         assert self.network is not None, "self.initialize_network must be called first"
#         self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
#                                          momentum=0.99, nesterov=True)
#         self.lr_scheduler = None
#
#     def setup_DA_params(self):
#         super().setup_DA_params()
#         self.data_aug_params["do_mirror"] = False
#         self.data_aug_params["do_elastic"] = True
#
#     def maybe_update_lr(self, epoch=None):
#         if epoch is None:
#             ep = self.epoch + 1
#         else:
#             ep = epoch
#         self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
#         self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
#
#     def run_iteration(self, src_data_generator, tar_data_generator, do_backprop=True, run_online_evaluation=False):
#         # Get Source&Target Data
#         src_data_dict = next(src_data_generator)
#         src_data = src_data_dict['data']
#         src_target = src_data_dict['target']
#
#         tar_data_dict = next(tar_data_generator)
#         tar_data = tar_data_dict['data']
#         tar_target = tar_data_dict['target']
#
#         # To Tensor
#         src_data = maybe_to_torch(src_data)
#         src_target = maybe_to_torch(src_target)
#         tar_data = maybe_to_torch(tar_data)
#         tar_target = maybe_to_torch(tar_target)
#
#         # To CUDA
#         if torch.cuda.is_available():
#             src_data = to_cuda(src_data, gpu_id=None)
#             src_target = to_cuda(src_target, gpu_id=None)
#             tar_data = to_cuda(tar_data, gpu_id=None)
#             tar_target = to_cuda(tar_target, gpu_id=None)
#
#         self.optimizer.zero_grad()
#
#         if self.fp16:
#             with autocast():
#                 src_output = self.network(src_data)
#                 tar_output = self.network(tar_data)
#                 del src_data, tar_data
#                 l = 0.5 * self.compute_loss(src_output, src_target) + self.compute_loss(tar_output, tar_target)
#
#             if do_backprop:
#                 self.amp_grad_scaler.scale(l).backward()
#                 self.amp_grad_scaler.unscale_(self.optimizer)
#                 torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
#                 self.amp_grad_scaler.step(self.optimizer)
#                 self.amp_grad_scaler.update()
#         else:
#             src_output = self.network(src_data)
#             tar_output = self.network(tar_data)
#             del src_data, tar_data
#             l = 0.5 * self.compute_loss(src_output, src_target) + self.compute_loss(tar_output, tar_target)
#
#             if do_backprop:
#                 l.backward()
#                 torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
#                 self.optimizer.step()
#
#         if run_online_evaluation:
#             self.run_online_evaluation(tar_output, tar_target)
#             self.run_another_online_evaluation(src_output, src_target)
#
#         del src_target, tar_target
#         return l.detach().cpu().numpy()


class nnUNetTrainerV2_FLARE_Pseudo_DDP(nnUNetTrainerV2_DDP_UDA):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice,
                         stage, unpack_data, deterministic, distribute_batch_size, fp16)

    def initialize_network(self):
        self.conv_per_stage = 3
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 100
        self.save_every = self.max_num_epochs // 10

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
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True,
                                        True,
                                        self.max_num_features, low_level=3)
        gen_net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        self.generator = Generic_GAN_UNet(self.num_input_channels, self.base_num_features, 1,
                                          len(gen_net_num_pool_op_kernel_sizes),
                                          2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                          net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
                                          InitWeights_He(1e-2),
                                          gen_net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True,
                                          False,
                                          self.max_num_features, low_level=3)
        self.discriminator = PatchDiscriminator(in_ch=1 + 14, base_num_features=self.base_num_features,
                                                pool_op_kernel_sizes=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                                                weightInitializer=InitWeights_He(1e-2))
        if torch.cuda.is_available():
            self.network.cuda()
            self.generator.cuda()
            self.discriminator.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        # Loss Function
        self.recon_loss = torch.nn.MSELoss()
        self.dis_loss = GAN_loss(gan_mode='bce')

        # Learning Rate
        # self.initial_lr = 1e-3
        self.gen_lr = 1e-4
        self.dis_lr = 2.5e-5

    def run_iteration(self, src_data_generator, tar_data_generator, do_backprop=True, run_online_evaluation=False):
        self.generator.eval()
        # Get Source&Target Data
        src_data_dict = next(src_data_generator)
        src_data = src_data_dict['data']
        src_target = src_data_dict['target']

        tar_data_dict = next(tar_data_generator)
        tar_data = tar_data_dict['data']
        tar_target = tar_data_dict['target']

        # To Tensor
        src_data = maybe_to_torch(src_data)
        src_target = maybe_to_torch(src_target)
        tar_data = maybe_to_torch(tar_data)
        tar_target = maybe_to_torch(tar_target)

        # To CUDA
        if torch.cuda.is_available():
            src_data = to_cuda(src_data, gpu_id=None)
            src_target = to_cuda(src_target, gpu_id=None)
            tar_data = to_cuda(tar_data, gpu_id=None)
            tar_target = to_cuda(tar_target, gpu_id=None)

        if self.fp16:
            with autocast():
                # Segmentation on MR
                mr_style_fea, mr_to_mr, mr_output = self.network(src_data, tar_data, tar_seg=True)
                # Generation: CT To MR
                with torch.no_grad():
                    ct_to_mr = self.generator(src_data, mr_style_fea[-1].detach())
                # Segmentation on CT2MR
                ct2mr_output = self.network(ct_to_mr, shallow_feat=False, last_feat=False)
                # Discriminator
                tar_data_norm = -1. + 2. * (tar_data - tar_data.min()) / (tar_data.max() - tar_data.min() + 1e-10)
                real_input = torch.cat((ct_to_mr.tanh().detach(), ct2mr_output[0].detach()), dim=1)
                fake_input = torch.cat((tar_data_norm, mr_output[0].detach()), dim=1)
                real_output = self.discriminator(real_input)
                fake_output = self.discriminator(fake_input)

                del src_data, tar_data
                # Loss for Segmentation Network and Generation Network
                seg_l_mr = self.compute_loss(mr_output, tar_target)
                seg_l_ct2mr = self.compute_loss(ct2mr_output, src_target)
                seg_l = seg_l_ct2mr + seg_l_mr
                recon_l = self.recon_loss(mr_to_mr.tanh(), tar_data_norm)
                # Loss for Discriminator Network
                dis_l = self.dis_loss(fake_output, target_is_real=False) + self.dis_loss(real_output, target_is_real=True)
                del real_input, fake_input, real_output, fake_output, mr_to_mr

            if do_backprop:
                # Update Discriminator Network
                self.dis_optimizer.zero_grad()
                self.amp_grad_scaler.scale(dis_l).backward()
                self.amp_grad_scaler.unscale_(self.dis_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 2.5)
                self.amp_grad_scaler.step(self.dis_optimizer)
                self.amp_grad_scaler.update()
                self.dis_optimizer.zero_grad()
                # Update Segmentation Network and Generator Network
                with autocast():
                    fake_input = torch.cat((tar_data_norm, mr_output[0]), dim=1)
                    fake_output = self.discriminator(fake_input)
                    gen_dis_l = self.dis_loss(fake_output, target_is_real=True)

                    l = seg_l + 0.1 * gen_dis_l + 0.1 * recon_l
                    self.optimizer.zero_grad()
                    self.amp_grad_scaler.scale(l).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
                    self.optimizer.zero_grad()
        else:
            # Segmentation on MR
            mr_style_fea, mr_to_mr, mr_output = self.network(src_data, tar_data, tar_seg=True)
            # Generation: CT To MR
            ct_to_mr = self.generator(src_data, mr_style_fea[-1].detach())
            # Segmentation on CT2MR
            ct2mr_output = self.network(ct_to_mr, shallow_feat=False, last_feat=False)
            # Discriminator
            tar_data_norm = -1. + 2. * (tar_data - tar_data.min()) / (tar_data.max() - tar_data.min() + 1e-10)
            real_input = torch.cat((ct_to_mr.tanh().detach(), ct2mr_output.detach()), dim=1)
            fake_input = torch.cat((tar_data_norm, mr_output.detach()), dim=1)

            del src_data, tar_data
            # Loss for Segmentation Network and Generation Network
            seg_l_mr = self.compute_loss(mr_output, tar_target)
            seg_l_ct2mr = self.compute_loss(ct2mr_output, src_target)
            seg_l = seg_l_ct2mr + seg_l_mr
            recon_l = self.recon_loss(mr_to_mr.tanh(), tar_data_norm)
            # Loss for Discriminator Network
            dis_l = self.dis_loss(fake_input, target_is_real=False) + self.dis_loss(real_input, target_is_real=True)
            del real_input, fake_input

            if do_backprop:
                # Update Discriminator Network
                self.dis_optimizer.zero_grad()
                dis_l.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 2.5)
                self.dis_optimizer.step()
                self.dis_optimizer.zero_grad()
                # Update Segmentation Network and Generator Network
                with autocast():
                    fake_input = torch.cat((tar_data_norm, mr_output), dim=1)
                    gen_dis_l = self.dis_loss(fake_input, target_is_real=True)

                    l = seg_l + 0.1 * gen_dis_l + 0.1 * recon_l
                    self.optimizer.zero_grad()
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(mr_output, tar_target)
            self.run_another_online_evaluation(ct2mr_output, src_target)
        del mr_output, ct2mr_output, src_target, tar_data_norm
        return l.detach().cpu().numpy()


class nnUNetTrainerV2_FLARE_Big_DDP(nnUNetTrainerV2_DDP_UDA):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice,
                         stage, unpack_data, deterministic, distribute_batch_size, fp16)

    def run_iteration(self, src_data_generator, tar_data_generator, do_backprop=True, run_online_evaluation=False):
        # Get Source&Target Data
        src_data_dict = next(src_data_generator)
        src_data = src_data_dict['data']
        src_target = src_data_dict['target']

        tar_data_dict = next(tar_data_generator)
        tar_data = tar_data_dict['data']

        # To Tensor
        src_data = maybe_to_torch(src_data)
        src_target = maybe_to_torch(src_target)
        tar_data = maybe_to_torch(tar_data)

        # To CUDA
        if torch.cuda.is_available():
            src_data = to_cuda(src_data, gpu_id=None)
            src_target = to_cuda(src_target, gpu_id=None)
            tar_data = to_cuda(tar_data, gpu_id=None)

        alpha = min(1., pow(((self.epoch + 1) / 250), 0.5))
        if self.epoch >= 500:
            ct_alpha = max(0.5, 1. - pow(((self.epoch + 1 - 500) / 250), 2))
        else:
            ct_alpha = 1

        if self.fp16:
            with autocast():
                # Segmentation on CT
                ct_output, ct_content_fea = self.network(src_data, shallow_feat=False, last_feat=True)
                mr_style_fea, mr_to_mr = self.network(src_data, tar_data)
                # Generation: CT To MR
                ct_to_mr = self.generator(src_data, mr_style_fea[-1].detach())
                # Segmentation on MR
                ct2mr_output, ct2mr_style_fea, ct2mr_content_fea = self.network(ct_to_mr, shallow_feat=True, last_feat=True)
                # Discriminator
                tar_data_norm = -1. + 2. * (tar_data - tar_data.min()) / (tar_data.max() - tar_data.min() + 1e-10)
                fake_mr = self.discriminator(ct_to_mr.tanh().detach())
                real_mr = self.discriminator(tar_data_norm)

                del tar_data
                # Loss for Segmentation Network and Generation Network
                seg_l_ct = self.compute_loss(ct_output, src_target)
                seg_l_ct2mr = self.compute_loss(ct2mr_output, src_target)
                seg_l = ct_alpha * seg_l_ct + alpha * seg_l_ct2mr
                recon_l = self.recon_loss(mr_to_mr.tanh(), tar_data_norm)
                consis_c_l = alpha * self.content_loss(ct2mr_content_fea, ct_content_fea.detach())
                consis_s_l = alpha * self.style_loss(ct2mr_style_fea, mr_style_fea)
                # Loss for Discriminator Network
                dis_l = self.dis_loss(fake_mr, target_is_real=False) + self.dis_loss(real_mr, target_is_real=True)
                del fake_mr, real_mr, tar_data_norm, ct2mr_content_fea, ct_content_fea

            if do_backprop:
                # Update Discriminator Network
                self.dis_optimizer.zero_grad()
                self.amp_grad_scaler.scale(dis_l).backward()
                self.amp_grad_scaler.unscale_(self.dis_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 2.5)
                self.amp_grad_scaler.step(self.dis_optimizer)
                self.amp_grad_scaler.update()
                self.dis_optimizer.zero_grad()
                # Update Segmentation Network and Generator Network
                with autocast():
                    fake_mr_gen = self.discriminator(ct_to_mr.tanh())
                    gen_dis_l = self.dis_loss(fake_mr_gen, target_is_real=True)
                if self.epoch < 25:
                    gen_l = gen_dis_l
                    self.gen_optimizer.zero_grad()
                    self.amp_grad_scaler.scale(gen_l).backward(retain_graph=True)
                    self.amp_grad_scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10)
                    self.amp_grad_scaler.step(self.gen_optimizer)
                    self.amp_grad_scaler.update()
                    self.gen_optimizer.zero_grad()
                    l = seg_l_ct + 0.1 * recon_l
                    self.optimizer.zero_grad()
                    self.amp_grad_scaler.scale(l).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    l = gen_dis_l + seg_l + 0.1 * consis_c_l + 0.01 * consis_s_l + 0.1 * recon_l
                    self.gen_optimizer.zero_grad(), self.optimizer.zero_grad()
                    self.amp_grad_scaler.scale(l).backward(retain_graph=True)
                    self.amp_grad_scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10)
                    self.amp_grad_scaler.step(self.gen_optimizer)
                    self.amp_grad_scaler.update()

                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
        else:
            # Segmentation on CT
            ct_output, ct_content_fea = self.network(src_data, shallow_feat=False, last_feat=True)
            mr_style_fea, mr_to_mr = self.network(src_data, tar_data)
            # Generation: CT To MR
            ct_to_mr = self.generator(src_data, tar_data)
            # Segmentation on MR
            ct2mr_output, ct2mr_style_fea, ct2mr_content_fea = self.network(ct_to_mr, shallow_feat=True, last_feat=True)
            # Discriminator
            tar_data_norm = -1. + 2. * (tar_data - tar_data.min()) / (tar_data.max() - tar_data.min())
            fake_mr = self.discriminator(ct_to_mr.tanh().detach())
            real_mr = self.discriminator(tar_data_norm)

            del tar_data
            # Loss for Segmentation Network
            seg_l_ct = alpha * self.compute_loss(ct_output, src_target)
            seg_l_ct2mr = (1. - alpha) * self.compute_loss(ct2mr_output, src_target)
            seg_l = seg_l_ct + seg_l_ct2mr
            seg_consis_l = (1. - alpha) * self.compute_seg_consis(ct2mr_output, ct_output)
            consis_c_l = (1. - alpha) * self.content_loss(ct2mr_content_fea, ct_content_fea.detach())
            recon_l = self.recon_loss(mr_to_mr.tanh(), tar_data_norm)
            # Loss for Generator Network
            consis_s_l = self.style_loss(ct2mr_style_fea, mr_style_fea.detach())
            # Loss for Discriminator Network
            dis_l = self.dis_loss(fake_mr, target_is_real=False) + self.dis_loss(real_mr, target_is_real=True)
            del fake_mr, real_mr, tar_data_norm, ct2mr_content_fea, ct_content_fea

            if do_backprop:
                # Update Segmentation Network
                l = seg_l + seg_consis_l + consis_c_l + 0.1 * recon_l
                l.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
                # Update Discriminator Network
                self.dis_optimizer.zero_grad()
                dis_l.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 2.5)
                self.dis_optimizer.step()
                # Update Generator Network
                fake_mr_gen = self.discriminator(ct_to_mr.tanh())
                gen_dis_l = self.dis_loss(fake_mr_gen, target_is_real=True)
                self.optimizer.zero_grad(), self.gen_optimizer.zero_grad()
                gen_l = consis_s_l + seg_l_ct2mr + 0.1 * gen_dis_l
                gen_l.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10)
                self.gen_optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(ct2mr_output, src_target)
            self.run_another_online_evaluation(ct_output, src_target)

        del ct2mr_output, src_target

        return l.detach().cpu().numpy(), seg_l.detach().cpu().numpy(), gen_dis_l.detach().cpu().numpy(), \
               dis_l.detach().cpu().numpy(), consis_s_l.detach().cpu().numpy(), consis_c_l.detach().cpu().numpy(), \
               ct_to_mr.tanh().detach().cpu().numpy().astype(np.float32), \
               src_data.detach().cpu().numpy().astype(np.float32), \
               alpha, src_data_dict['properties'], tar_data_dict['properties']
