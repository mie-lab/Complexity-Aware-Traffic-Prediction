
# Complexity-Aware Traffic Prediction 


This is the code accompanying the paper _Intrinsic Complexity of Deep Learning-based City-wide Traffic Prediction_  
The paper is available as a pre-print ([link to pre-print](http://dx.doi.org/10.13140/RG.2.2.29917.18405/1))
## Dependencies

Ubuntu: 
```bash
conda create --name pyenv python=3.11
conda activate pyenv
pip install -r requirements_frozen.txt
```
If GPU not available, comment out these lines:
```bash
# #######################
# nvidia-cublas-cu11==11.11.3.6
# nvidia-cudnn-cu11==8.6.0.163
```
- Ensure that gpustat is not installed after successful installation of tensorflow. For GPU-usage while running, please 
use `nvidia-smi` instead


## Data
- The traffic data can be downloaded by following the instructions provided on the competition page for [Traffic4Cast2022](https://github.com/iarai/NeurIPS2022-traffic4cast)


Commit id for reproducing results from the paper: 
1c7c302e68276a5c8a61be7bbefca5d36b871ec6


## Demo 
For instructions on how to use the proposed metrics for a new dataset, please follow the detailed documentation using demo data at [Demo_CATP.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP.ipynb)


## Citing
If you find this code useful in your research, please consider citing:
```
@unknown{unknown,
author = {Kumar, Nishant and Martin, Henry and Martin, Raubal},
year = {2023},
month = {12},
pages = {},
title = {Intrinsic Complexity: Quantifying Task Complexity For Deep Learning-Based City-Wide Traffic Prediction},
doi = {10.13140/RG.2.2.29917.18405/1}
}
```
