
# Complexity-Aware Traffic Prediction 


This is the code accompanying the paper _Enhancing deep learning-based city-wide traffic prediction pipelines through complexity analysis_  

## Dependencies

Ubuntu: 
```bash
conda create --name pyenv python=3.11
conda activate pyenv
pip install -r requirements_frozen.txt
```
If GPU is not available, comment out these lines:
```bash
# #######################
# nvidia-cublas-cu11==11.11.3.6
# nvidia-cudnn-cu11==8.6.0.163
```
- Please ensure that `gpustat` is not installed. For GPU-usage while the script is running, please use `nvidia-smi` instead


## Data
- The traffic data can be downloaded by following the instructions provided on the competition page for [Traffic4Cast2022](https://github.com/iarai/NeurIPS2022-traffic4cast)


Commit id for reproducing results from the paper: 
1c7c302e68276a5c8a61be7bbefca5d36b871ec6


## Demo 
- For instructions on how to use the proposed metrics for a new dataset, please follow the detailed documentation using demo data at [Demo_CATP.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP.ipynb)
- Sometimes, the github preview is slow and it might not load the Demo Jupyter notebook correctly due to figures in the outputs. If that happens, the Demo notebook with outputs removed can be viewed at [Demo_CATP_without_images.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP_outputs_deleted.ipynb)
- A quick preview of the demo in `.html` format can be downloaded from [Demo_CATP_html_format.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP_html_format.html) and viewed using firefox.
- The demo script runs well on [google colab](https://colab.research.google.com) and and colab tensorflow version for the demo script that has been tested as of December 2023 is as shown below:
  ```python
  import tensorflow as tf
  tf.__version__
  # 2.15.0
  ```


[//]: # (## Citing)

[//]: # (If you find this code useful in your research, please consider citing:)

[//]: # (```)

[//]: # (@unknown{kumar2024complexityb,)

[//]: # (author = {Kumar, Nishant and Martin, Henry and Martin, Raubal},)

[//]: # (year = {2023},)

[//]: # (month = {12},)

[//]: # (pages = {},)

[//]: # (title = {Intrinsic Complexity: Quantifying Task Complexity For Deep Learning-Based City-Wide Traffic Prediction},)

[//]: # (doi = {10.13140/RG.2.2.29917.18405/1})

[//]: # (})

[//]: # (```)

Details regarding the implementation-level considerations for the four key formulations from the paper can be found in the Jupyter notebook: [Demo_CATP.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP.ipynb). The four key formulations are reproduced here for convenience.

**Equation (7)** in paper: The _criticality of a data point (sample)_ is given by Sample Complexity (SC) of a data point; given a model _f_
```math
SC({X}|f,\mathcal{D}) = \sum_{{X_j} \in \mathbb{T}_{X}} \left(d_{f({X_j})} - r_{X}\right) \cdot 1_{d_{f({X_j})}>r_{X}} 
```
**Equation (8)** in paper: Model Complexity (MC) of a model is defined as the mean of sample complexities (_criticalities_) for all data points.
```math
MC(f| \mathcal{D}) = \frac{1}{N}\sum_{k=1}^{N} SC({X}_k|f, \mathcal{D})
```
**Equation (9)** in paper: Intrinsic Sample Complexity (ISC) of a data point; given a prediction task

```math
        ISC({X}|\mathcal{D}) =\sum_{{X}_j \in \mathbb{T}x} \left(d_{{Y}_j} -r_{X}\right) \cdot 1_{d_{{Y}_j}>r_{X}}
```
**Equation (11)** in paper:  Intrinsic Complexity (IC) of a prediction task
```math
        IC (Task) = \frac{1}{N}\sum_{k=1}^{N} ISC({X}_k|{D})
```
 


