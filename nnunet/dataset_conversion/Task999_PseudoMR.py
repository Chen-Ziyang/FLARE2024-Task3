import os
import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


if __name__ == '__main__':
    downloaded_flare_dir = '/data1/userdisk0/zychen/flare2024'

    target_dataset_id = 999
    target_dataset_name = f'Task{target_dataset_id:3.0f}_PseudoMR'
    out_base = join(nnUNet_raw_data, target_dataset_name)
    print(out_base)

    imagesTr = join(nnUNet_raw_data, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw_data, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # copy images
    source = join(downloaded_flare_dir, 'imagesTr', 'Register_MR')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        if not os.path.isfile(join(imagesTr, s)):
            shutil.copy(join(source, s), join(imagesTr, s))

    source = join(downloaded_flare_dir, 'imagesTr', 'HardSample')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        if not os.path.isfile(join(imagesTr, s)):
            shutil.copy(join(source, s), join(imagesTr, s))

    source = join(downloaded_flare_dir, 'imagesTr', 'Amos')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        if not os.path.isfile(join(imagesTr, s)):
            shutil.copy(join(source, s), join(imagesTr, s))

    # copy labels
    source = join('/data1/userdisk0/zychen/RESULTS_FOLDER/AMOS')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        if not os.path.isfile(join(labelsTr, s)):
            shutil.copy(join(source, s), join(labelsTr, s))

    phases = ['C+A', 'C+Delay', 'C+V', 'C-pre', 'DWI', 'InPhase', 'OutPhase', 'T2WI']
    source = join('/data1/userdisk0/zychen/RESULTS_FOLDER/LLD-MMRI-Pseudo')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        if not os.path.isfile(join(labelsTr, s)):
            for phase in phases:
                shutil.copy(join(source, s), join(labelsTr, s[:-7] + '_' + phase + '.nii.gz'))

    train_identifiers = get_identifiers_from_splitted_files(imagesTr)

    json_dict = {}
    json_dict['name'] = target_dataset_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = "hands off!"
    json_dict['release'] = '0.0'
    json_dict['modality'] = {0: "MR"}
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
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in train_identifiers]
    save_json(json_dict, join(out_base, 'dataset.json'), sort_keys=True)

