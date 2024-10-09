import os
import shutil
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def convert_labels(seg: np.ndarray):
    new_seg = np.copy(seg)
    new_seg[seg > 13] = 0
    return new_seg


def load_convert_labels(filename, input_folder, output_folder):
    lbl = sitk.ReadImage(join(input_folder, filename))
    lbl_array = sitk.GetArrayFromImage(lbl)
    new_lbl_array = convert_labels(lbl_array)
    new_lbl = sitk.GetImageFromArray(new_lbl_array)
    new_lbl.CopyInformation(lbl)
    sitk.WriteImage(new_lbl, join(output_folder, filename.replace('_0000.nii.gz', '.nii.gz')))


if __name__ == '__main__':
    downloaded_flare_dir = '/data1/userdisk0/zychen/flare2024'

    target_dataset_id = 666
    target_dataset_name = f'Task{target_dataset_id:3.0f}_FLARE2024Challenge'
    out_base = join(nnUNet_raw_data, target_dataset_name)
    print(out_base)

    imagesTr = join(nnUNet_raw_data, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw_data, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # copy images
    source_CT = join(downloaded_flare_dir, 'imagesTr', 'CT')
    source_CT_files = nifti_files(source_CT, join=False)
    for s in source_CT_files:
        if not os.path.isfile(join(imagesTr, s)):
            shutil.copy(join(source_CT, s), join(imagesTr, s))

    # copy labels
    source_CT = join(downloaded_flare_dir, 'labelsTr')
    source_CT_files = nifti_files(source_CT, join=False)
    for s in source_CT_files:
        if not os.path.isfile(join(labelsTr, s.replace('_0000.nii.gz', '.nii.gz'))):
            load_convert_labels(s, source_CT, labelsTr)

    train_identifiers = get_identifiers_from_splitted_files(imagesTr)

    json_dict = {}
    json_dict['name'] = target_dataset_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = "hands off!"
    json_dict['release'] = '0.0'
    json_dict['modality'] = {0: "CT"}
    json_dict['labels'] = {
                              0: "background",
                              1: "liver",
                              2: "right kidney",
                              3: "spleen",
                              4: "pancreas",
                              5: "aorta",
                              6: "inferior vena cava",
                              7: "right adrenal gland",
                              8: "left adrenal gland",
                              9: "gallbladder",
                              10: "esophagus",
                              11: "stomach",
                              12: "duodenum",
                              13: "left kidney",
                          }

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['training'] = [{'image': "./imagesTr/CT/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in train_identifiers]
    save_json(json_dict, join(out_base, 'dataset.json'), sort_keys=True)

