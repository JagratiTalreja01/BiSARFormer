# BiSARFormerGAN: Dual-Band SAR-to-Optical Translation via Transformer-Enhanced GAN
BiSARFormer is a deep learning framework for translating Synthetic Aperture Radar (SAR) imagery into perceptually realistic optical imagery using dual polarization inputs (VV &amp; VH) via local cross-attention fusion, and a Transformer-based generator within a GAN setup to capture complementary scattering patterns. 


The code is built on [MT_GAN (PyTorch)](https://github.com/NUAA-RS/MT_GAN) and tested on Ubuntu 20.04.6 environment (Python3.10.13, PyTorch >= 1.1.0) with NVIDIA RTX A4000 with 16GB RAM. 
## Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Train](#train)
4. [Test](#test)
5. [Results](#results)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Introduction

This repository contains the implementation of BiSARFormerGAN, a dual-band SAR-to-Optical translation framework designed for flood assessment. The architecture introduces dual-stem processing for Sentinel-1 VV and VH polarizations, followed by Local Cross-Attention Fusion (LCAF) to exploit their complementary scattering properties. To balance local spatial detail and global contextual modeling, the network integrates CNN backbones with Transformer modules inside a GAN framework, enhanced by SE-gated skip connections.

Trained on the DeepFlood dataset, BiSARFormerGAN generates optical-like imagery that preserves fine-grained structures, improves perceptual quality, and supports reliable flood mapping and disaster response.

![Generator BiSARFormerGAN](./Figures/3FIGURE.PDF)

![LCAF (Local Cross Attention Fusion)](./Figures/2FIGURE.PDF)

![LCAF (Discriminator BiSARFormer)](./Figures/9FIGURE.PDF)


Key Highlights:

Dual-Polarization Processing: Independent stems for SAR VV and VH bands, extracting complementary scattering features.

Local Cross-Attention Fusion (LCAF): Adaptive fusion mechanism that leverages interactions between VV and VH polarizations for richer representation.

Hybrid GAN–Transformer Framework: Combines the perceptual realism of GANs with the global contextual modeling power of Transformers.

SE-Gated Skip Connections: Noise-aware skip links that selectively pass useful features while suppressing SAR-induced artifacts.

Efficient and Stable Training: Incorporates Residual Swin Transformer Blocks and multi-loss optimization (L1, SSIM, perceptual, adversarial) for balanced realism and accuracy.

Superior Performance: Outperforms state-of-the-art SAR-to-Optical translation models on the DeepFlood dataset in PSNR, SSIM, and LPIPS, while producing interpretable optical-like outputs for real flood events.

BiSARFormerGAN bridges SAR and optical modalities, improving the usability of SAR imagery for flood mapping, disaster response, and geospatial analysis.

## Dependencies
* Python 3.10.13
* PyTorch >= 1.1.0
* CUDA 12.2
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Train
### Prepare training data 

1. Download DEEPFLOOD Dataset which includes co-registered Sentinel-1 SAR (VV, VH) and Sentinel-2 optical imagery, along with UAV references and auxiliary layers (NDWI, slope, DTM, flood masks). from [DEEPFLOOD dataset][(https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339)].

2. Use SAR_VH, SAR_VV  for Dual-Polarization input and and UAV tiles for Target Optical

3. Create train, test and validation set 70%, 15% & 15%

4. Specify '--dir_data' based on the images path. 

For more information, please refer to [MT_GAN (PyTorch)](https://github.com/NUAA-RS/MT_GAN).

### Begin to train

Cd to 'src', run the following script to train models.

 **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model dhtcun --scale 2 --patch_size 96 --save DHTCUN_x2 --data_train DIV2K
    ```
## Test
### Quick start
1. Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)


Cd to 'src', run the following scripts.

 **Example command is in the file 'demo.sh'.**

    ```bash
    
    # Example X2 SR
    python main.py --dir_data ../../ --model dhtcun  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only 
    ```

## Results
### Visual Patches

![Urban100x4](./Figures/Urbanx4.PNG)

![BSD100x8](./Figures/BSDx8.PNG)

![Urnan100x8](./Figures/Urbanx8.PNG)

![Manga109x8](./Figures/Mangax8.PNG)

### Quantitative Results

![Number of Parameters](./Figures/Parameters.PNG)

![Execution Time](./Figures/Execution_Time.PNG)

![Space Complexity](./Figures/Space_complexity.PNG)

![Time Complexity](./Figures/Time_complexity.PNG)

![Multi-Adds](./Figures/Multi-Addds.PNG)

![PSNR Convergence](./Figures/Convergence.PNG)

For more Quantitative Results please read the paper [[Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10648606&tag=1)

## Citation
If you find the code helpful in your research or work, please cite the following papers.
```
@ARTICLE{10648606,
  author={Talreja, Jagrati and Aramvith, Supavadee and Onoye, Takao},
  journal={IEEE Access}, 
  title={DHTCUN: Deep Hybrid Transformer CNN U Network for Single-Image Super-Resolution}, 
  year={2024},
  volume={12},
  number={},
  pages={122624-122641},
  keywords={Transformers;Superresolution;Convolutional neural networks;Noise measurement;Computational modeling;Noise measurement;Image reconstruction;CNN;enhanced spatial attention;single-image super-resolution;Transformer},
  doi={10.1109/ACCESS.2024.3450300}}

```

## Acknowledgements
This code is built on [HNCT (PyTorch)](https://github.com/lhjthp/HNCT/tree/main) and [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
