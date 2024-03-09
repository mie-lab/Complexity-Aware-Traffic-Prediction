
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
- Please ensure that `gpustat` is not installed. For GPU-usage while the script is running, please use `nvidia-smi` instead


## Data
- The traffic data can be downloaded by following the instructions provided on the competition page for [Traffic4Cast2022](https://github.com/iarai/NeurIPS2022-traffic4cast)


Commit id for reproducing results from the paper: 
1c7c302e68276a5c8a61be7bbefca5d36b871ec6


## Demo 
- For instructions on how to use the proposed metrics for a new dataset, please follow the detailed documentation using demo data at [Demo_CATP.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP.ipynb)
- Sometimes, the github preview is slow and it might not load the Demo Jupyter notebook correctly due to figures in the outputs. If that happens, the Demo notebook with outputs removed can be viewed at [Demo_CATP_without_images.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP_outputs_deleted.ipynb)
- A quick preview of the demo in `.html` format can be downloaded from [Demo_CATP_html_format.ipynb](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP_html_format.html) and viewed using firefox. In case of trouble rendering the .html files in GitHub viewer, the PDF files can be used to preview the implementation details (contents same as .html files) from [PDF_with_outputs](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP.pdf) and [PDF_with_outputs](https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/main/CATP/Demo_CATP_outputs_deleted.pdf)
- The demo script runs well on [google colab](https://colab.research.google.com) and and colab tensorflow version for the demo script that has been tested as of December 2023 is as shown below:
  ```python
  import tensorflow as tf
  tf.__version__
  # 2.15.0
  ```


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
