# An Orchestration Learning Framework for Ultrasound Imaging: Prompt-Guided Hyper-Perception and Attention-Matching Downstream Synchronization

This repository provides the official PyTorch implementation for our work published in **Medical Image Analysis, 2025**. The framework introduces:

- **Prompt-Guided Hyper-Perception** for incorporating prior domain knowledge via learnable prompts.
- **Attention-Matching Downstream Synchronization** to seamlessly transfer knowledge across segmentation and classification tasks.
- Support for diverse ultrasound datasets with both segmentation and classification annotations (**$M^2-US$ dataset**).
- Distributed training and inference pipelines based on the Swin Transformer backbone.

For more details, please refer to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001860) ([temporary free link, expires on July 17, 2025](https://authors.elsevier.com/c/1lAXd4rfPmLew4)) and [Project Page](https://zehui-lin.github.io/PerceptGuide/),

> [**An orchestration learning framework for ultrasound imaging: Prompt-guided hyper-perception and attention-matching Downstream Synchronization**](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001860)<br/>
  Zehui Lin, Shuo Li, Shanshan Wang, Zhifan Gao, Yue Sun, Chan-Tong Lam, Xindi Hu, Xin Yang, Dong Ni, and Tao Tan. <b>Medical Image Analysis</b>, 2025.


## Installation
1. Clone the repository
```bash
git clone https://github.com/Zehui-Lin/PerceptGuide
cd PerceptGuide
```
2. Create a conda environment
```bash
conda create -n PerceptGuide python=3.10
conda activate PerceptGuide
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```

## Data
Organize your data directory with `classification` and `segmentation` sub-folders. Each sub-folder should contain a `config.yaml` file and train/val/test lists:
```
data
├── classification
│   └── DatasetA
│       ├── 0
│       ├── 1
│       ├── config.yaml
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── segmentation
    └── DatasetB
        ├── imgs
        ├── masks
        ├── config.yaml
        ├── train.txt
        ├── val.txt
        └── test.txt
```
Use the examples provided in the codebase as a reference when preparing new datasets.

## Dataset Licensing & Redistribution

The repository bundles several ultrasound datasets. Their licenses and redistribution conditions are listed below. You can download the preprocessed datasets which allow for redistribution from [here](https://github.com/Zehui-Lin/PerceptGuide/releases/tag/v1.0.0).

| Dataset | License | Redistribution | Access |
|---------|---------|---------------|--------|
| Appendix | CC BY-NC 4.0 | Included in repo | [link](https://zenodo.org/records/7669442) |
| BUS-BRA | CC BY 4.0 | Included in repo | [link](https://zenodo.org/records/8231412) |
| BUSIS | CC BY 4.0 | Included in repo | [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9025635/) |
| UDIAT | Private License | Not redistributable | [link](https://ieeexplore.ieee.org/abstract/document/8003418) |
| CCAU | CC BY 4.0 | Included in repo | [link](https://data.mendeley.com/datasets/d4xt63mgjm/1) |
| CUBS | CC BY 4.0 | Included in repo | [link](https://data.mendeley.com/datasets/fpv535fss7/1) |
| DDTI | Unspecified License | License unclear | [link](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) |
| TN3K | Unspecified License | License unclear | [link](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) |
| EchoNet-Dynamic | Private License | Not redistributable | [link](https://echonet.github.io/dynamic/index.html) |
| Fatty-Liver | CC BY 4.0 | Included in repo | [link](https://zenodo.org/records/1009146) |
| Fetal_HC | CC BY 4.0 | Included in repo | [link](https://zenodo.org/records/1327317) |
| MMOTU | CC BY 4.0 | Included in repo | [link](https://figshare.com/articles/dataset/_zip/25058690?file=44222642) |
| kidneyUS | CC BY-NC-SA | Included in repo | [link](https://rsingla.ca/kidneyUS/) |
| BUSI | CC0 Public Domain | Included in repo | [link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| HMC-QU | CC BY 4.0 | Included in repo | [link](https://www.kaggle.com/aysendegerli/hmcqu-dataset) |
| TG3K | Unspecified License | License unclear | [link](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) |

**Notes**

- **Private-license datasets** (UDIAT, EchoNet-Dynamic) cannot be redistributed here; please request access through the provided links.  
- **Unspecified/unclear-license datasets** (TN3K, TG3K, DDTI) may have redistribution restrictions. Download them directly from the source or contact the data owners for permission.

## Training
We employ `torch.distributed` for multi-GPU training (single GPU is also supported):
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 omni_train.py --output_dir exp_out/trial_1 --prompt
```

## Testing
For evaluation, run:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 omni_test.py --output_dir exp_out/trial_1 --prompt
```

## Pretrained Weights
Download the Swin Transformer backbone and place it in `pretrained_ckpt/`:
* [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

The folder structure should look like:
```
pretrained_ckpt
└── swin_tiny_patch4_window7_224.pth
```
## Checkpoint

You can download the pre-trained checkpoints from the [release](https://github.com/Zehui-Lin/PerceptGuide/releases/tag/v1.0.0) pages.


## Citation
If you find this project helpful, please consider citing:
```bibtex
@article{lin2025orchestration,
  title={An orchestration learning framework for ultrasound imaging: Prompt-guided hyper-perception and attention-matching Downstream Synchronization},
  author={Lin, Zehui and Li, Shuo and Wang, Shanshan and Gao, Zhifan and Sun, Yue and Lam, Chan-Tong and Hu, Xindi and Yang, Xin and Ni, Dong and Tan, Tao},
  journal={Medical Image Analysis},
  pages={103639},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements
This repository is built upon the [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) codebase. We thank the authors for making their work publicly available.
