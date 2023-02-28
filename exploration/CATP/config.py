import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DEBUG = False

DATA_FOLDER_LOCAL = "/Users/nishant/Documents/GitHub/CTP/exploration"
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
RESULTS_FOLDER = os.path.join(DATA_FOLDER, "results")

######################## complexity class params #########################
cx_max_dist = 4000
cx_method = "fractional"
cx_tnl = 8
cx_N = 20
cx_re_compute_y_thresh = True  # if not recomputing, then we use the value of y_thresh from the variable "cx_y_thresh"
cx_y_thresh = 1250


######################## Datagen class params #########################
dg_debug = False


######################## ConvLSTM class params #########################
cl_dataloader_workers = 8
cl_early_stopping_patience = 4
cl_percentage_of_train_data = 0.0025  # can be reduced for fast tryouts
cl_loss_func = "mse"
cl_batch_size = 2 # 32
cl_epochs = 2
cl_n_depth = 3
