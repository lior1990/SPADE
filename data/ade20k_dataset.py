"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=26)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'validation' if opt.phase == 'test' else 'training'

        all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []
        for p in all_images:
            if phase not in p:
                continue
            if "images" in p:
                image_paths.append(p)
            elif "annotations" in p:
                label_paths.append(p)

        instance_paths = []  # don't use instance map for ade20k

        if len(label_paths) != len(image_paths) and phase == "validation":
            images_folder = os.path.split(image_paths[0])[0]
            image_paths = []
            for label_path in label_paths:
                corresponding_img_name = os.path.basename(label_path)
                corresponding_img_name = f"{'_'.join(corresponding_img_name.split('_')[:-2])}.jpg"
                image_paths.append(os.path.join(images_folder,corresponding_img_name))

        return label_paths, image_paths, instance_paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    # def postprocess(self, input_dict):
    #     label = input_dict['label']
    #     label = label - 1
    #     label[label == -1] = self.opt.label_nc
