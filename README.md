# ECEA: Extensible Co-Existing Attention for Few-Shot Object Detection

Config files coming soon~

![image](https://github.com/zhimengXin/ECEA/assets/162425451/cd519983-439c-43e2-ad4b-489b5e7a7f3f)

## Quick Start

```
Linux with Python >= 3.8
PyTorch >= 2.1.1 & torchvision that matches the PyTorch version.
CUDA 11.7, 11.8
GCC >= 4.9
```
## Install Detectron2
```
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{torch_version}/{cuda_version}/index.html
```
## Install other requirements.

```
python3 -m pip install -r requirements.txt
```

## Take coco as an example to train ECEA
'''
bash run_coco_base.sh

bash run_coco_fsod.sh / bash run_coco_gfsod.sh

'''
## Citing

This repo is developed based on FSCE, DeFRCN and Detectron2. Please check them for more details and features.

If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@ARTICLE{10558758,
  author={Xin, Zhimeng and Wu, Tianxu and Chen, Shiming and Zou, Yixiong and Shao, Ling and You, Xinge},
  journal={IEEE Transactions on Image Processing}, 
  title={ECEA: Extensible Co-Existing Attention for Few-Shot Object Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Training;Detectors;Object detection;Feature extraction;Task analysis;Semantics;Adaptation models;Few-shot object detection;extensible attention;co-existing regions},
  doi={10.1109/TIP.2024.3411771}}
```
