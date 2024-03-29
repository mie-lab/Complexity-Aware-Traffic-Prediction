"""

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
  
"""

import datetime
import glob
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RUNNING_IC_TEMP = False


DEBUG = False

delete_processed_files = True

cur_dir = os.getcwd()
if cur_dir.split("/")[1] == "home":
    if cur_dir.split("/")[2] == "kumar":
        running_on = "homepc"
    elif cur_dir.split("/")[2] == "niskumar":
        running_on = "server"
elif cur_dir.split("/")[1] == "Users":
    running_on = "maclocal"
else:
    raise Exception("Unknown computer; please check config file")


if running_on == "server":
    HOME_FOLDER = "/home/niskumar/CATP/CATP"
    DATA_FOLDER = "/home/niskumar/CATP/NeurIPS2022-traffic4cast"

elif running_on == "maclocal":
    HOME_FOLDER = "/Users/nishant/Documents/GitHub/CTP/CATP"
    DATA_FOLDER = "/Users/nishant/Downloads/NeurIPS2022-traffic4cast"
else:
    raise Exception("HOME_FOLDER and DATA_FOLDER not specified; please check config file")


max_norm_value = 1

INTERMEDIATE_FOLDER = os.path.join(HOME_FOLDER, "intermediate_folder")
RESULTS_FOLDER = os.path.join(DATA_FOLDER, "intermediate_folder")

######################## complexity class params #########################
# cx_max_dist = 2500
# cx_method = "fractional"
# cx_tnl = 8
# if running_on=="server":
#     cx_N = 300
# else:
#     cx_N = 5
cx_debug = True
if running_on == "server":
    cx_sample_whole_data = -1 # 1500
    cx_sample_single_point = 200  # no longer being used
    cx_batch_size = 32

elif running_on == "maclocal":
    cx_sample_whole_data = -1
    cx_sample_single_point = 40  # no longer being used
    cx_batch_size = 32

cx_sampling_enabled = False
cx_delete_files_after_running = True
cx_range_day_scan = range(-3, 4)
cx_range_t_band_scan = range(-4, 5)
cx_spatial_cx_PM_dist_enabled = False
cx_post_model_loading_from_saved_val_error_plots_temporal = False
cx_post_model_loading_from_saved_val_error_plots_spatial_save_spatial_npy = False

######################## Datagen class params #########################
dg_debug = False
dg_debug_each_data_sample = False

######################## ConvLSTM class params #########################
cl_model_save_epoch_end = True
cl_model_save_train_end = True
cl_early_stopping_patience = -1  # -1 implies no early stopping
cl_tensorboard = False

if running_on == "server":
    cl_percentage_of_train_data = 1  # can be reduced for fast tryouts
    cl_batch_size = 32
    cl_prediction_batch_size = 32 
    cl_dataloader_workers = 32
    cl_epochs = 30
elif running_on == "maclocal":
    cl_percentage_of_train_data = 0.002  # can be reduced for fast tryouts
    cl_batch_size = 3
    cl_prediction_batch_size = 32
    cl_dataloader_workers = 8
    cl_epochs = 2

cl_loss_func = "mse"  # "mse"
cl_n_depth = 3
cl_during_training_CSR_enabled_epoch_end = True


######################### Dimensions for experiments ####################
if running_on == "server":
    city_list = [
        "madrid",
        "melbourne",
        "london",
    ]  # , "madrid", "MELBOURNE"]  # all are converted to lower case later on
    scales = list(range(25, 106, 10))  # [25, 200, 250, 150, 225, 50, 125, 75, 100, 175]
    i_o_lengths = list(range(1, 9))
    pred_horiz = list(range(1, 9))

    city_list_def = ["London"]
    scales_def = [55]
    i_o_lengths_def = [4]
    pred_horiz_def = [1]


elif running_on == "maclocal":
    # city_list = ["LonDON"]  # all are converted to lower case later on
    # scales_def = [45]
    # i_o_lengths_def = [1]
    # pred_horiz_def = [1]
    # scales = [8, 16]  # , 16] # [1, 8, 16, 32, 64, 128, 256]
    # i_o_lengths = [1] # , 8]  # [1, 2, 3, 4, 5, 6, 7, 8]
    # pred_horiz = [1] # , 8]  # [1, 2, 3, 4, 5, 6, 7, 8]
    city_list = [
        "london",
        "melbourne",
        "madrid",
        
    ]  # , "madrid", "MELBOURNE"]  # all are converted to lower case later on
    scales = list(range(25, 106, 10))  # [25, 200, 250, 150, 225, 50, 125, 75, 100, 175]
    i_o_lengths = list(range(1, 9))
    pred_horiz = list(range(1, 9))

    city_list_def = ["madrid"]
    scales_def = [55]
    i_o_lengths_def = [4]
    pred_horiz_def = [1]



########################### Computing prediction error ###########################
# filename: predict_errors_and_csr_saved_model.py
val_err_choose_last_epoch = False
val_error_log_csv_folder_path = "/Users/nishant/Downloads/val_csv_85"


DATA_START_DATE = {
    "london": datetime.date(2019, 7, 1),
    "madrid": datetime.date(2021, 6, 1),
    "melbourne": datetime.date(2020, 6, 1),
}

DATA_END_DATE = {
    "london": datetime.date(2020, 1, 31),
    "madrid": datetime.date(2021, 12, 31),
    "melbourne": datetime.date(2020, 12, 30),
}

val_folder_name = "val_data_all_cities"
train_folder_name = "train_data_all_cities"

if not os.path.exists(os.path.join(DATA_FOLDER, train_folder_name)):
    os.mkdir(os.path.join(DATA_FOLDER, train_folder_name))
if not os.path.exists(os.path.join(DATA_FOLDER, val_folder_name)):
    os.mkdir(os.path.join(DATA_FOLDER, val_folder_name))

TRAINING_DATA_FOLDER = os.path.join(DATA_FOLDER, train_folder_name)
VALIDATION_DATA_FOLDER = os.path.join(DATA_FOLDER, val_folder_name)

if running_on == "server":
    cutoff_day_number_train = int(30 * 3.5)
    start_day_number_val = int(30 * 3.5)

elif running_on == "maclocal":
    cutoff_day_number_train = int(30 * 3.5)
    start_day_number_val = int(30 * 3.5)

# ensure no overlap between train and validation data
assert start_day_number_val >= cutoff_day_number_train
