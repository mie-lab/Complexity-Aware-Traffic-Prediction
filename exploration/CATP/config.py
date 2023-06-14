import datetime
import glob
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    HOME_FOLDER = "/home/niskumar/NeurIPS2022-traffic4cast/exploration/CATP"
    DATA_FOLDER = "/home/niskumar/NeurIPS2022-traffic4cast"

elif running_on == "maclocal":
    HOME_FOLDER = "/Users/nishant/Documents/GitHub/CTP/exploration/CATP"
    DATA_FOLDER = "/Users/nishant/Downloads/NeurIPS2022-traffic4cast"

else:
    raise Exception("HOME_FOLDER and DATA_FOLDER not specified; please check config file")


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
# cx_re_compute_y_thresh = False  # if not recomputing, then we use the value of y_thresh from the variable "cx_y_thresh"
cx_debug = True
if running_on == "server":
    cx_sample_whole_data = 1500
    cx_sample_single_point = 200  # no longer being used
    cx_batch_size = 14

elif running_on == "maclocal":
    cx_sample_whole_data = 2
    cx_sample_single_point = 40  # no longer being used
    cx_batch_size = 2

cx_delete_files_after_running = True
cx_range_day_scan = range(-3, 4)
cx_range_t_band_scan = range(-4, 5)
cx_spatial_cx_dist_enabled = True


######################## Datagen class params #########################
dg_debug = False
dg_debug_each_data_sample = False

######################## ConvLSTM class params #########################
cl_model_save_epoch_end = True
cl_model_save_train_end = True
cl_early_stopping_patience = 5  # -1 implies no early stopping
cl_tensorboard = False
cl_thresh = 750

if running_on == "server":
    cl_percentage_of_train_data = 1  # can be reduced for fast tryouts
    cl_batch_size = 32
    cl_dataloader_workers = 32
    cl_epochs = 30
elif running_on == "maclocal":
    cl_percentage_of_train_data = 1  # can be reduced for fast tryouts
    cl_batch_size = 3
    cl_dataloader_workers = 4
    cl_epochs = 20

cl_loss_func = "mse"  # "mse"
cl_n_depth = 3
cl_during_training_CSR_enabled_epoch_end = True
cl_during_training_CSR_enabled_batch_end = True
cl_during_training_CSR_enabled_train_end = True


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

    city_list_def = ["London"]
    scales_def = [55]
    i_o_lengths_def = [4]
    pred_horiz_def = [1]


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
val_folder_name_sparse = "val_data_all_cities_sparse"
train_folder_name_sparse = "train_data_all_cities_sparse"

if not os.path.exists(os.path.join(DATA_FOLDER, train_folder_name)):
    os.mkdir(os.path.join(DATA_FOLDER, train_folder_name))
if not os.path.exists(os.path.join(DATA_FOLDER, val_folder_name)):
    os.mkdir(os.path.join(DATA_FOLDER, val_folder_name))
if not os.path.exists(os.path.join(DATA_FOLDER, train_folder_name_sparse)):
    os.mkdir(os.path.join(DATA_FOLDER, train_folder_name_sparse))
if not os.path.exists(os.path.join(DATA_FOLDER, val_folder_name_sparse)):
    os.mkdir(os.path.join(DATA_FOLDER, val_folder_name_sparse))

TRAINING_DATA_FOLDER = os.path.join(DATA_FOLDER, train_folder_name)
VALIDATION_DATA_FOLDER = os.path.join(DATA_FOLDER, val_folder_name)
TRAINING_DATA_FOLDER_SPARSE = os.path.join(DATA_FOLDER, train_folder_name_sparse)
VALIDATION_DATA_FOLDER_SPARSE = os.path.join(DATA_FOLDER, val_folder_name_sparse)

if running_on == "server":
    cutoff_day_number_train = int(30 * 3.5)
    start_day_number_val = int(30 * 3.5)

elif running_on == "maclocal":
    cutoff_day_number_train = int(30 * 3.5)
    start_day_number_val = int(30 * 3.5)

# ensure no overlap between train and validation data
assert start_day_number_val >= cutoff_day_number_train
