# Solution of Team NPUSAIIP for MICCAI 2024 Challenge FLARE Task3
**Rethinking nnU-Net for Cross-Modality Unsupervised Domain Adaptation in Abdominal Organ Segmentation** \
*Ziyang Chen, Xiaoyu Bai, Zhisong Wang, Yiwen Ye, Yongsheng Pan*, and Yong Xia* \

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).
Part of the code is revised from the Pytorch implementation of [https://github.com/Ziyan-Huang/FLARE22](https://github.com/Ziyan-Huang/FLARE22/).

This repository provides the solution of team npusaiip for [MICCAI 2024 Challenge FLARE Task3](https://www.codabench.org/competitions/2296/). 

The details of our method are described in our [paper](https://openreview.net/forum?id=dI5SeoVkV5). 

Our trained model is available at Our trained model is available at [RESULTS_FOLDER](./RESULTS_FOLDER)

## Environments and Requirements:
Install nnU-Net [1] as below. You should meet the requirements of nnUNet, our method does not need any additional requirements. For more details, please refer to https://github.com/MIC-DKFZ/nnUNet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
Copy the nnunet folder in this repo to your environment.

## 1. UDA Phase: Train the Uni-Net
### 1.1. Prepare Labeled CT and Unlabeled MR
Run the [nnunet/dataset_conversion/Task666_FLARE2024Challenge.py](./nnunet/dataset_conversion/Task666_FLARE2024Challenge.py) file to prepare CT data.

Run the [nnunet/dataset_conversion/Task777_MR.py](./nnunet/dataset_conversion/Task777_MR.py) file to prepare MR data.

### 1.2. Conduct Automatic Preprocessing using nnUNet
Here we do not use the default setting.
```
nnUNet_plan_and_preprocess -t 666 -pl3d ExperimentPlanner3D_FLARE24Big -pl2d None
nnUNet_plan_and_preprocess -t 777 -pl3d ExperimentPlanner3D_FLARE24Big -pl2d None
```

### 1.3. Train the Uni-Net using DDP
Set ```self.MR_task='Task777_MR'``` in the [nnunet/training/network_training/nnUNetTrainerUDA.py](./nnunet/training/network_training/nnUNetTrainerUDA.py) file
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=2 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_FLARE_Big_DDP 666 all -p nnUNetPlansFLARE24Big --dbs
```

## 2. Pseudo-Labeling Phase
### 2.1. Registration for LLD-MMRI
We first register scans from eight modalities per patient following [UAE](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/) and replace the original data by registered data.

### 2.2. Generate Pseudo Labels for Unlabeled MR Data using trained Uni-Net
Copy a new result folder without 'DDP', and then run the following codes:
```
python utils/modify_pkl_Uni-Net.py
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER -t 666 -tr nnUNetTrainerV2_FLARE_Big -m 3d_fullres -p nnUNetPlansFLARE24Big -f all --all_in_gpu True 
```
After predicting the results, we ensemble the predictions of four modalities (i.e., C+A, C+Delay, C+V, and C-pre) in LLD-MMRI dataset by running the [utils/merge_label_LLD-MMRI.py](./utils/merge_label_LLD-MMRI.py) file.

### 2.3. Prepare MR with Pseudo Labels
Run the [nnunet/dataset_conversion/Task999_PseudoMR.py](./nnunet/dataset_conversion/Task999_PseudoMR.py) file to prepare MR data with pseudo labels.

### 2.4. Conduct Automatic Preprocessing using nnUNet
```
nnUNet_plan_and_preprocess -t 999 -pl3d ExperimentPlanner3D_FLARE24Big -pl2d None
```
Run the [utils/remove_no_label.py](./utils/remove_no_label.py) file to remove samples without labels for AMOS dataset.

### 2.5. Fine-tune the Uni-Net using DDP
Set ```self.MR_task='Task999_PseudoMR'``` in the [nnunet/training/network_training/nnUNetTrainerUDA.py](./nnunet/training/network_training/nnUNetTrainerUDA.py) file
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=2 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_FLARE_Pseudo_DDP 666 all -p nnUNetPlansFLARE24Big --dbs
```

### 2.6. Train the MR-Net
```
python utils/modify_batch_size.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=2 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_FLARE_MR_DDP 999 all -p nnUNetPlansFLARE24Big --dbs
```

## 3. Final Phase: Train the LW-Net
### 3.1. Generate Pseudo Labels for Unlabeled MR Data using fine-tuned Uni-Net and trained MR-Net
Copy a new result folder without 'DDP', and then run the following codes:
**Fine-tuned Uni-Net**
```
python utils/modify_pkl_ft_Uni-Net.py
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER -t 666 -tr nnUNetTrainerV2_FLARE_Pseudo -m 3d_fullres -p nnUNetPlansFLARE24Big -f all --all_in_gpu True 
```
**Trained MR-Net**
```
python utils/modify_pkl_MR-Net.py
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER -t 999 -tr nnUNetTrainerV2_FLARE_Pseudo -m 3d_fullres -p nnUNetPlansFLARE24Big -f all --all_in_gpu True 
```
After predicting the results, we ensemble the predictions of these two models by running the [utils/pseudo_label.py](./utils/pseudo_label.py) file.

### 3.2. Conduct Automatic Preprocessing using nnUNet
Re-preprocess the MR data with new pseudo labels.
```
nnUNet_plan_and_preprocess -t 999 -pl3d ExperimentPlanner3D_FLARE24Small -pl2d None
```
Run the [utils/remove_no_label.py](./utils/remove_no_label.py) file to remove samples without labels for AMOS dataset.

### 3.2. Train the LW-Net
```
python utils/modify_batch_size.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=2 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_FLARE_Small 999 all -p nnUNetPlansFLARE24Big --dbs
```

## 4. Perform Efficient Inference with LW-Net
```
python utils/modify_pkl_LW-Net.py
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 999 -p nnUNetPlansFLARE24Small -m 3d_fullres -tr nnUNetTrainerV2_FLARE_Small -f all --mode fastest --disable_tta
```
