import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from typing import Sequence

# prompt info dict
# task prompt
position_prompt_list = [
    "segmentation",
    "classification",
]
# position prompt
position_prompt_dict = {
    "MMOTU": "ovary",
    "BUSI": "breast",
    "TN3K": "thyroid",
    "CUBS": "carotid",
    "BUS-BRA": "breast",
    "Appendix": "appendix",
    "Fatty-Liver": "liver",
    "UDIAT": "breast",
    "DDTI": "thyroid",
    "HMC-QU": "cardiac",
    "Fetal_HC": "head",
    "BUSIS": "breast",
    "CCAU": "carotid",
    "kidneyUS_capsule": "kidney",
    "TG3K": "thyroid",
    "EchoNet-Dynamic": "cardiac",
}

# type prompt
type_prompt_dict = {
    "DDTI": "tumor",
    "MMOTU": "tumor",
    "BUSI": "tumor",
    "HMC-QU": "organ",
    "TN3K": "tumor",
    "Fetal_HC": "organ",
    "BUSIS": "tumor",
    "CCAU": "organ",
    "CUBS": "organ",
    "BUS-BRA": "tumor",
    "kidneyUS_capsule": "organ",
    "TG3K": "organ",
    "EchoNet-Dynamic": "organ",
    "UDIAT": "tumor",
    "Appendix": "organ",
    "Fatty-Liver": "organ",
}

# mode prompt
available_mode_prompt_list = [
    "DDTI",
    "MMOTU",
    "BUSI",
    "HMC-QU",
    "TN3K",
    "Fetal_HC",
    "BUSIS",
    "CCAU",
    "CUBS",
    "BUS-BRA",
    "kidneyUS_capsule",
    "TG3K",
    "EchoNet-Dynamic",
    "UDIAT",
]


# prompt one-hot
# organ prompt
position_prompt_one_hot_dict = {
    "ovary": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "breast": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "thyroid": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "carotid": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "liver": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "cardiac": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "head": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "kidney": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "leg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "indis": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


# task prompt
task_prompt_one_hot_dict = {"segmentation": [1, 0], "classification": [0, 1]}
# type prompt
type_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}
# mode prompt
mode_prompt_one_hot_dict = {
    "whole": [1, 0, 0],
    "local": [0, 1, 0],
    "location": [0, 0, 1],
}


def list_add_prefix(txt_path, prefix_1, prefix_2):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    if prefix_2 is not None:
        return [os.path.join(prefix_1, prefix_2, line.strip("\n")) for line in lines]
    else:
        return [os.path.join(prefix_1, line.strip("\n")) for line in lines]


class WeightedRandomSamplerDDP(DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        data_set,
        weights: Sequence[float],
        num_replicas: int,
        rank: int,
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(num_samples)
            )
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got " "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank :: self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def get_localImg(image, mask):
    if mask.sum() == 0:
        return image, mask, False
    else:
        x, y, w, h = cv2.boundingRect(mask)
        length = max(w, h)
        local_image = image[y : y + length, x : x + length, :]
        local_mask = mask[y : y + length, x : x + length]
        return local_image, local_mask, True


def get_enhanceImg(image, mask):
    if mask.sum() == 0:
        return image, False
    else:
        # mask[mask > 0] = 255
        bin_mark = mask.copy()
        bin_mark[bin_mark > 0] = 1
        enhance_image = image + (np.expand_dims(bin_mark, axis=2) * 0.1).astype("uint8")
        return enhance_image, True


def get_sample_list(seg_dataset, cls_dataset, base_dir, split):
    return_list = []
    for dataset in seg_dataset:
        dataset_return_list = list_add_prefix(
            os.path.join(base_dir, "segmentation", dataset, split + ".txt"), "segmentation/" + dataset, "imgs"
        )

        if dataset == "BUSI":
            dataset_return_list = [sample for sample in dataset_return_list if not "normal" in sample]

        return_list.extend(dataset_return_list)
    for dataset in cls_dataset:
        dataset_return_list = list_add_prefix(
            os.path.join(base_dir, "classification", dataset, split + ".txt"), "classification/" + dataset, None
        )

        if dataset == "BUSI":
            dataset_return_list = [sample for sample in dataset_return_list if not "normal" in sample]

        return_list.extend(dataset_return_list)
    return return_list


def get_subset_len(seg_dataset, cls_dataset, base_dir, split):
    subset_len = []
    for dataset in seg_dataset:
        dataset_subset_len = len(
            list_add_prefix(os.path.join(base_dir, "segmentation", dataset, split + ".txt"), dataset, "imgs")
        )

        if dataset == "BUSI":
            dataset_return_list = list_add_prefix(
                os.path.join(base_dir, "segmentation", dataset, split + ".txt"), dataset, "imgs"
            )
            dataset_subset_len = len([sample for sample in dataset_return_list if not "normal" in sample])

        subset_len.append(dataset_subset_len)
    for dataset in cls_dataset:
        dataset_subset_len = len(
            list_add_prefix(os.path.join(base_dir, "classification", dataset, split + ".txt"), dataset, None)
        )

        if dataset == "BUSI":
            dataset_return_list = list_add_prefix(
                os.path.join(base_dir, "classification", dataset, split + ".txt"), dataset, None
            )
            dataset_subset_len = len([sample for sample in dataset_return_list if not "normal" in sample])

        subset_len.append(dataset_subset_len)
    return subset_len


class USdatasetOmni(Dataset):
    def __init__(self, base_dir, split, transform=None, prompt=False):
        self.data_dir = base_dir
        self.split = split
        self.transform = transform
        self.prompt = prompt
        self.sample_list = []
        self.subset_len = []
        self.seg_use_dataset = [
            "DDTI",
            "MMOTU",
            "TN3K",
            "Fetal_HC",
            "BUSIS",
            "CCAU",
            "BUS-BRA",
            "kidneyUS_capsule",
            "EchoNet-Dynamic",
            "UDIAT",
        ]
        self.cls_use_dataset = ["TN3K", "CUBS", "BUS-BRA", "Appendix", "Fatty-Liver", "UDIAT"]

        self.sample_list = get_sample_list(self.seg_use_dataset, self.cls_use_dataset, base_dir, split)
        self.subset_len = get_subset_len(self.seg_use_dataset, self.cls_use_dataset, base_dir, split)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip("\n")
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)

        img_task = self.sample_list[idx].strip("\n").split("/")[0]
        dataset_name = img_name.split("/")[1]

        if "HMC-QU" in img_name:
            image = cv2.resize(image, (256, 256))

        if img_task == "segmentation":
            label_path = os.path.join(self.data_dir, img_name).replace("imgs", "masks")
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if "HMC-QU" in img_name:
                label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

            label_info = open(os.path.join(self.data_dir, "segmentation", dataset_name, "config.yaml")).readlines()
            label_info_list = [info.strip().split(":") for info in label_info]
            for single_label_info in label_info_list:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                label[label == label_value_in_image] = label_index
            # --------------------------------------------------------------------#
            if not self.prompt:
                sample = {"image": image / 255.0, "label": label}
            else:
                if random.random() > 0.5:
                    image, label, notZeroMask = get_localImg(image, label)
                    sample = {"image": image / 255.0, "label": label}
                    if notZeroMask:
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["local"]
                    else:
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                else:
                    sample = {"image": image / 255.0, "label": label}
                    sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
            # --------------------------------------------------------------------#

        elif img_task == "classification":
            label = int(img_name.split("/")[-2])

        if img_task == "classification":
            # --------------------------------------------------------------------#
            if not self.prompt:
                sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
            else:
                if dataset_name in available_mode_prompt_list:
                    random_number = random.random()
                    mask_path = os.path.join(
                        self.data_dir,
                        "segmentation",
                        "/".join([img_name.split("/")[1], "masks", img_name.split("/")[3]]),
                    )
                    mask_path = mask_path.replace(".jpg", ".png").replace(".JPG", ".png")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if random_number < 0.3:
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                    elif random_number < 0.6:
                        image, mask, notZeroMask = get_localImg(image, mask)
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        if notZeroMask:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["local"]
                        else:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                    else:
                        image, notZeroMask = get_enhanceImg(image, mask)
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        if notZeroMask:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["location"]
                        else:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                else:
                    sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                    sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
            # --------------------------------------------------------------------#

        if self.transform:
            sample = self.transform(sample)
        # label
        if img_task == "segmentation":
            sample["1dLabel"] = torch.from_numpy(np.array(0)).float()
            sample["2dLabel"] = sample["label"].float()
        elif img_task == "classification":
            sample["1dLabel"] = torch.from_numpy(np.array(label)).float()
            sample["2dLabel"] = torch.from_numpy(np.zeros(sample["image"].shape[1:3])).float()

        sample["case_name"] = self.sample_list[idx].strip("\n")
        sample["type_prompt"] = type_prompt_one_hot_dict[type_prompt_dict[dataset_name]]
        sample["position_prompt"] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        sample["task"] = img_task
        sample["task_prompt"] = task_prompt_one_hot_dict[img_task]
        sample["dataset_name"] = dataset_name

        return sample
