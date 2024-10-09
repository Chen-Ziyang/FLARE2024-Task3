import os, shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


### For AMOS
save_root = '/data1/userdisk0/zychen/RESULTS_FOLDER/AMOS'
root1 = '/data1/userdisk0/zychen/RESULTS_FOLDER/Task666_Pseudo_PseudoLabel/AMOS'
root2 = '/data1/userdisk0/zychen/RESULTS_FOLDER/Task999_Pseudo_PseudoLabel/AMOS'

if not os.path.exists(save_root):
    os.makedirs(save_root)

u_dict = {}
seg_masks = [i for i in os.listdir(root1) if i.endswith('nii.gz')]
for file_name in seg_masks:
    itk_image_ite1 = sitk.ReadImage(os.path.join(root1, file_name))
    data_npy_ite1 = sitk.GetArrayFromImage(itk_image_ite1)
    itk_image_ite2 = sitk.ReadImage(os.path.join(root2, file_name))
    data_npy_ite2 = sitk.GetArrayFromImage(itk_image_ite2)
    uncertainty2 = np.sum(data_npy_ite1 != data_npy_ite2)/np.sum(data_npy_ite2>0)
    u = uncertainty2
    print(file_name, u)
    u_dict[file_name] = u
u_order = sorted(u_dict.items(),key=lambda x:x[1],reverse=True)
print(u_order)
for k, v in u_dict.items():
    if v < 0.5 and not np.isnan(v) and not np.isinf(v): # 0.25 for LLD-MMRI
        print(k)
        shutil.copy(os.path.join(root2, k), os.path.join(save_root, k))



### For LLD-MMRI
save_root = '/data1/userdisk0/zychen/RESULTS_FOLDER/LLD-MMRI'
root1 = '/data1/userdisk0/zychen/RESULTS_FOLDER/Task666_Pseudo_PseudoLabel/LLD-MMRI'
root2 = '/data1/userdisk0/zychen/RESULTS_FOLDER/Task999_Pseudo_PseudoLabel/LLD-MMRI'

if not os.path.exists(save_root):
    os.makedirs(save_root)

u_dict = {}
seg_masks = [i for i in os.listdir(root1) if i.endswith('nii.gz')]
for file_name in seg_masks:
    itk_image_ite1 = sitk.ReadImage(os.path.join(root1, file_name))
    data_npy_ite1 = sitk.GetArrayFromImage(itk_image_ite1)
    itk_image_ite2 = sitk.ReadImage(os.path.join(root2, file_name))
    data_npy_ite2 = sitk.GetArrayFromImage(itk_image_ite2)
    uncertainty2 = np.sum(data_npy_ite1 != data_npy_ite2)/np.sum(data_npy_ite2>0)
    u = uncertainty2
    print(file_name, u)
    u_dict[file_name] = u
u_order = sorted(u_dict.items(),key=lambda x:x[1],reverse=True)
print(u_order)
for k, v in u_dict.items():
    if v < 0.25 and not np.isnan(v) and not np.isinf(v):
        print(k)
        shutil.copy(os.path.join(root2, k), os.path.join(save_root, k))

