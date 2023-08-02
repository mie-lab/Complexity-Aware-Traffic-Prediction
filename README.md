
# Complexity-Aware Traffic Prediction 


This is the code accompanying the paper _Intrinsic Complexity of Deep Learning-based City-wide Traffic Prediction_

## Dependencies
- Please use `pip install -r requirements.txt` to install all dependencies, including the ones needed for reading files from the competition repository for Traffic4Cast2022.


## Config file
The config file (located at `CATP/config.py`) contains the key parameters that can be changed in the current code version. The default values are coherent with the results in our paper. 

- The key dimensions that can be changed to redefine experiments are present in the config file. The variable names inside the config file are self-explanatory. The most important ones are described below:
  https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/8e80dd4c29db6a598377d9210955c6c2bdf327d7/CATP/config.py#L112-L114
- The home folder should be specified in the config file as shown below:
  https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/8e80dd4c29db6a598377d9210955c6c2bdf327d7/CATP/config.py#L24-L26
- The train-validation split can be modified as shown below:
  https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/8e80dd4c29db6a598377d9210955c6c2bdf327d7/CATP/config.py#L185
- If the trained model is to be saved after each epoch, the parameters should be set accordingly, as shown below:
  https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/8e80dd4c29db6a598377d9210955c6c2bdf327d7/CATP/config.py#L72
- The GPU number can be changed as shown below:
  https://github.com/mie-lab/Complexity-Aware-Traffic-Prediction/blob/8e80dd4c29db6a598377d9210955c6c2bdf327d7/CATP/config.py#L6
  Please note that multiple GPUs are not supported with the current code version. It will be released later on as needed.
  

## Data
- The traffic data can be downloaded by following the instructions provided on the competition page for [Traffic4Cast2022](https://github.com/iarai/NeurIPS2022-traffic4cast)
