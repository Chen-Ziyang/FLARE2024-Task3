import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

save_root = '/data1/userdisk0/zychen/RESULTS_FOLDER/LLD-MMRI-Pseudo'
seg_root = '/data1/userdisk0/zychen/RESULTS_FOLDER/LLD-MMRI'
LLD_root = '/data1/userdisk0/zychen/flare2024/registered-1_new'

if not os.path.exists(save_root):
    os.makedirs(save_root)

save_patients = os.listdir(save_root)
LLD_patients = os.listdir(LLD_root)
seg_masks = os.listdir(seg_root)
phases = ['C+A', 'C+Delay', 'C+V', 'C-pre']
print('All Patients:', len(LLD_patients))
print(len(save_patients), 'patients have been saved')
LLD_patients = [item for item in LLD_patients if item+'.nii.gz' not in save_patients]
print(len(LLD_patients), 'patients still need to be saved')

for patient in tqdm(LLD_patients):
    one_hot_s = []
    for phase in phases:
        mask = sitk.ReadImage(os.path.join(seg_root, patient + '_' + phase + '.nii.gz'))
        mask_npy = sitk.GetArrayFromImage(mask)

        one_hot = np.eye(13+1)[mask_npy]
        one_hot_s.append(one_hot)
    one_hot_s = np.stack(one_hot_s).sum(0)
    one_hot_s[one_hot_s <= 2] = 0
    one_hot_s[one_hot_s > 2] = 1
    one_hot_s = np.argmax(one_hot_s, axis=-1)

    pseudo = sitk.GetImageFromArray(one_hot_s.astype(np.uint8))
    pseudo.SetSpacing(mask.GetSpacing())
    pseudo.SetOrigin(mask.GetOrigin())
    pseudo.SetDirection(mask.GetDirection())

    sitk.WriteImage(pseudo, os.path.join(save_root, patient + '.nii.gz'))
    print(os.path.join(save_root, patient + '.nii.gz'))
