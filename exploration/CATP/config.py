import os

DEBUG = True
HOME_FOLDER = "/Users/nishant/Documents/GitHub/CTP/exploration/CATP"
DATA_FOLDER = "/Users/nishant/Documents/GitHub/CTP/exploration"

INTERMEDIATE_FOLDER = os.path.join(HOME_FOLDER, "intermediate_folder")

######################## complexity class params #########################
cx_max_dist = 4000
cx_method = "fractional"
cx_tnl = 8
cx_N = 200
cx_re_compute_y_thresh = False  # if not recomputing, then we use the value of y_thresh from the variable "cx_y_thresh"
cx_y_thresh = 1250


######################## Datagen class params #########################
dg_debug = False


######################## ConvLSTM class params #########################
cl_dataloader_workers = 3
cl_early_stopping_patience = 3
cl_percentage_of_train_data = 0.005  # for fast tryouts
cl_loss_func = "mse"
cl_batch_size = 2
cl_epochs = 5
cl_n_depth = 3
