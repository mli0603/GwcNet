import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from natsort import natsorted


class SceneFlowDatset(Dataset):
    def __init__(self, datapath, training):
        self.datapath = datapath
        self.training = training

        directory = os.path.join(datapath, 'frame_finalpass', 'TEST')
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                            os.path.isdir(os.path.join(sub_folder, seq))]

        self.left_filenames = []
        for seq_folder in seq_folders:
            self.left_filenames += [os.path.join(seq_folder, 'left', img) for img in
                                    os.listdir(os.path.join(seq_folder, 'left'))]

        self.left_filenames = natsorted(self.left_filenames)
        self.right_filenames = [left.replace('left', 'right') for left in self.left_filenames]
        self.disp_filenames = [left.replace('frame_finalpass', 'disparity').replace('.png', '.pfm') for left in
                               self.left_filenames]

        directory = os.path.join(datapath, 'occlusion', 'TEST', 'left')
        occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        self.occ_data = natsorted(occ_data)

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # occ
        occ = np.array(Image.open(self.occ_data[index])).astype(np.bool)
        disparity[occ] = 0.0

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            top_pad = 544 - h  # pad to 544
            right_pad = 960 - w

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": top_pad,
                    "right_pad": right_pad}
