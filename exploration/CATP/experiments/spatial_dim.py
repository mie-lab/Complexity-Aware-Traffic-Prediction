import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config
from models.ConvLSTM import ConvLSTM
from smartprint import smartprint as sprint
from preprocessing.ProcessRaw import ProcessRaw


class SpatialDim:
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
                model = self.model_class(
                    cityname,
                    io_length,
                    pred_horiz,
                    scale,
                    log_dir,
                    shape,
                    validation_csv_file,
                )

                model.train()
            else:
                raise (Exception("Wrong model class; Probably not implemented!"))


if __name__ == "__main__":

    ############ If testing with single values
    # cityname = "london"
    # io_length = 8
    # pred_horiz = 4
    # scale = 8
    # th = SpatialDim(cityname, io_length, pred_horiz, scale, model_class_str="ConvLSTM"
    #                 ).run_experiments()  # 1, 16, 32, 64, 128, 252

    ############ Running all experiments together
    for cityname in config.city_list:
        for io_length in config.i_o_lengths:
            for pred_horiz in config.pred_horiz:
                for scale in config.scales:
                    sprint(cityname, io_length, pred_horiz, scale)
                    th = SpatialDim(
                        cityname, io_length, pred_horiz, scale, model_class_str="ConvLSTM"
                    ).run_experiments()  # 1, 16, 32, 64, 128, 252
