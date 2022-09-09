<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.ulaval.ca/en/" target="_blank">
    <img src="https://ssc.ca/sites/default/files/logo-ulaval-reseaux-sociaux.jpg" alt="Logo" width="280" height="100">
  </a>

  <h3 align="center">Drone-based Thermographic Defect Detection</h3>
  <h4 align="center">Drone-based Thermographic Defect Detection of Large Specimens using Unsupervised Segmentation</h4>

  <br/>
  <br/>
  </p>
</p>

## TODO
- [ ] Thermal Data Labeling
- [x] Implement the project structure
- Implement metrics:
  - [x] Confusion Matrix
  - [x] mIoU
  - [x] RMSE
  - [x] PSNR
  - [x] Directed Hausdorff Distance
  - [x] FSSIM
  - [x] SSIM
  - [x] ISSM
  - [x] UIQ
  - [x] SAM
  - [x] SRE
- Models:
  - Deep Learning Methods:
    - [x] Kanezaki2018
    - [x] Wonjik2020
    - [x] WNet
    - [x] UNET with RESNET18 encoder
    - [ ] DFR
    - [ ] AutoEncoder with SSIM loss (https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-)
    - [ ] Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals
  - Classical Methods:
    - [ ] Normalized Cut
    - [x] DBSCAN 
    - [x] K-Mean
    - [x] Mean Shift 
    - [x] Graph Cut
    - [ ] Spectral clustering (https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py)
- Segmentor
  - [x] Iterative Segmentor
  - [ ] Non-Iterative Segmentor
- Additional Layers
  - [x] CRF Layer

## Datasets
- [ ] Simulated Heated Plates
- [ ] CRF, Glass, and other plates
- [ ] Pipe inspection
- [ ] Aerospace component inspection

## Experiment
- [ ] Apply the segmentation methods on the collected dataset.
- [ ] Compare network with CRF layer and without CRF layer.

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
  * [Team](#team)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [Trained Model](#trained-model)
* [Dataset](#dataset)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Docker](#docker)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About The Project
**Drone-based Thermographic Defect Detection** : Drone-based Thermographic Defect Detection of Large Specimens using Unsupervised Segmentation.

**ABSTRACT** - .........

### Team
* Designed and Developed by: Parham Nooralishahi, PhD. candidate @ Computer and Electrical Engineering Department, Université Laval
* Supervisor: Professor Xavier Maldague
* Industrial Supervisor: Dr. Fernando Lopez
* Program: Doctorate in Electrical Engineering
* University: Université Laval

### Built With
* [PyTorch](https://pytorch.org/)
* [Pillow](https://pypi.org/project/Pillow/)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Contact
Parham Nooralishahi - parham.nooralishahi.1@ulaval.ca | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>

## Acknowledgements
Thanks to **TORNGATS** for providing the required equipment and support for performing the experiment and requirement analysis.

