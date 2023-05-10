import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DEBUG = False

DATA_FOLDER_LOCAL = "/Users/nishant/Downloads/NeurIPS2022-traffic4cast"
HOME_FOLDER_LOCAL = "/Users/nishant/Documents/GitHub/CTP/exploration/CATP"
DATA_FOLDER_SERVER = "/home/niskumar/NeurIPS2022-traffic4cast/exploration"
HOME_FOLDER_SERVER = "/home/niskumar/NeurIPS2022-traffic4cast/exploration/CATP"

cur_dir = os.getcwd()
if cur_dir.split("/")[1] == "home":
    HOME_FOLDER = HOME_FOLDER_SERVER
    DATA_FOLDER = DATA_FOLDER_SERVER
elif cur_dir.split("/")[1] == "Users":
    HOME_FOLDER = HOME_FOLDER_LOCAL
    DATA_FOLDER = DATA_FOLDER_LOCAL

INTERMEDIATE_FOLDER = os.path.join(HOME_FOLDER, "intermediate_folder")
RESULTS_FOLDER = os.path.join(DATA_FOLDER, "results/latest_2")

######################## complexity class params #########################
cx_max_dist = 2500
cx_method = "fractional"
cx_tnl = 8
if cur_dir.split("/")[1] == "home":
    cx_N = 300
else:
    cx_N = 5
cx_re_compute_y_thresh = False  # if not recomputing, then we use the value of y_thresh from the variable "cx_y_thresh"


######################## Datagen class params #########################
dg_debug = True


######################## ConvLSTM class params #########################
cl_model_save = False
cl_early_stopping_patience = -1
cl_tensorboard = False

if cur_dir.split("/")[1] == "home":
    cl_percentage_of_train_data = 0.1  # can be reduced for fast tryouts
    cl_batch_size = 128
    cl_dataloader_workers = 16
    cl_epochs = 20
elif cur_dir.split("/")[1] == "Users":
    cl_percentage_of_train_data = 0.05  # can be reduced for fast tryouts
    cl_batch_size = 2
    cl_dataloader_workers = 4
    cl_epochs = 2

cl_loss_func = "mse"
cl_n_depth = 3


######################### Dimensions for experiments ####################
scales = [8]  # , 16] # [1, 8, 16, 32, 64, 128, 256]
i_o_lengths = [4, 8]  # [1, 2, 3, 4, 5, 6, 7, 8]
pred_horiz = [4, 8]  # [1, 2, 3, 4, 5, 6, 7, 8]
city_list = ["LonDON", "madrid", "MELBOURNE"]  # all are converted to lower case later on

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

cutoff_day_number_train = 20
start_day_number_val = 190

# ensure no overlap between train and validation data
assert start_day_number_val >= cutoff_day_number_train
