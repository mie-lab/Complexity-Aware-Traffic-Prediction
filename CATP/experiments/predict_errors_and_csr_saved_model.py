import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config
from models.ConvLSTM import ConvLSTM
from smartprint import smartprint as sprint
from preprocessing.ProcessRaw import ProcessRaw
import glob
from tqdm import tqdm
import pandas as pd


class Experiment_Epoch:
    def __init__(self, cityname, io_length, pred_horiz, scale, model_class_str):
        assert model_class_str in ["ConvLSTM"]  # others not implemented yet

        self.cityname, self.io_length, self.pred_horiz, self.scale = cityname, io_length, pred_horiz, scale

        self.params_for_DL_model = []

        self.postfix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)

        self.params_for_DL_model.append(
            (
                self.cityname,
                self.io_length,
                self.pred_horiz,
                self.scale,
                "logs_dir_" + self.postfix + ".csv",
                "val_csv_" + self.postfix + ".csv",
            )
        )
        if model_class_str == "ConvLSTM":
            self.model_class_str = "ConvLSTM"
            self.model_class = ConvLSTM

    def run_experiments(self):
        for cityname, io_length, pred_horiz, scale, log_dir, validation_csv_file in self.params_for_DL_model:
            shape = (2, 1, self.scale, self.scale, 1)

            sprint(cityname, io_length, pred_horiz, scale, log_dir, shape, validation_csv_file)
            if self.model_class_str == "ConvLSTM":
                if config.val_err_choose_last_epoch:
                    epoch = 1
                    while os.path.exists(os.path.join(config.INTERMEDIATE_FOLDER,
                            os.path.basename(os.path.normpath(self.postfix)) + "_epoch_" + str(epoch) + ".h5")):
                        epoch += 1
                    epoch -= 1

                else:
                    assert config.val_error_log_csv_folder_path is not None
                    prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)
                    filename = os.path.join(config.val_error_log_csv_folder_path,  "val_csv_"+ prefix + ".csv")

                    df = pd.read_csv(filename)

                    # Find the epoch with the lowest val_loss
                    lowest_epoch = df['epoch'][df['val_loss'].idxmin()]
                    sprint (prefix)
                    print("The epoch with the lowest val_loss is: ", lowest_epoch)

                    epoch = lowest_epoch

                sprint (cityname, io_length, pred_horiz, scale, epoch)

                model = self.model_class(
                    cityname,
                    io_length,
                    pred_horiz,
                    scale,
                    log_dir,
                    shape,
                    validation_csv_file,
                    saved_model_filename=os.path.join(config.INTERMEDIATE_FOLDER,
                        os.path.basename(os.path.normpath(self.postfix)) + "_epoch_" + str(epoch) + ".h5")
                )

                # model.model = model.create_model_small_epochs()
                print(model.model.summary())
                model.model = model.train()
                print ("Model loaded successfully; epoch@", epoch)
            else:
                raise (Exception("Wrong model class; not implemented!"))


if __name__ == "__main__":
    for io_length in [4]:  # config.i_o_lengths_def:
        for pred_horiz in [1]:  # config.pred_horiz_def:
            for scale in [85]:  # config.scales_def:
                for cityname in config.city_list:
                    sprint(cityname, io_length, pred_horiz, scale)

                    # for a in [
                    #     glob.glob(config.TRAINING_DATA_FOLDER + "/*"),
                    #     glob.glob(config.VALIDATION_DATA_FOLDER + "/*"),
                    # ]:
                    #     for i in tqdm(range(0, len(a), 1), desc="deleting old temp file"):
                    #         string_list = (
                    #             str(a[i : i + 1])
                    #             .replace(",", "")
                    #             .replace("[", "")
                    #             .replace("]", "")
                    #             .strip()
                    #             .replace('"', "")
                    #             .replace("'", "")
                    #         )
                    #         os.system("rm -rf " + string_list)

                    obj = ProcessRaw(
                        cityname=cityname, i_o_length=io_length, prediction_horizon=pred_horiz, grid_size=scale
                    )

                    # try:
                    th = Experiment_Epoch(
                        cityname, io_length, pred_horiz, scale, model_class_str="ConvLSTM"
                    ).run_experiments()  # 1, 16, 32, 64, 128, 252
                    # except Exception as e:
                    #     # raise Exception(e)
                    #     # Need to use raise Exception only when debugging, otherwise, we can just ignore and move on -
                    #     # so that the runs for multiple scenarios are completed even if one fails.
                    #     print("ERROR in ", cityname, io_length, pred_horiz, scale, "Exiting; No results for this case")
                    #     continue

                    # obj.clean_intermediate_files()
                    # ProcessRaw.clean_intermediate_files(cityname, io_length, pred_horiz, scale)
