import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config
from models.ConvLSTM import ConvLSTM
from smartprint import smartprint as sprint


class TemporalIOLength:
    def __init__(self, range_of_io_ranges, base_folder_name_i_h_o, model_class_str):
        """
        :param range_of_io_ranges: list of i/o values
        :param base_folder_name_i_h_o: must follow this format training_data_8_4_8
        """
        assert model_class_str in ["ConvLSTM"]  # others not implemented yet
        self.range_of_io_ranges = range_of_io_ranges
        _, _, self.i, self.h, self.o = base_folder_name_i_h_o.split("_")
        assert self.i == self.o
        self.list_of_folders = []
        for io in self.range_of_io_ranges:
            self.list_of_folders.append(
                (
                    "training_data_" + str(io) + "_" + self.h + "_" + str(io),
                    "validation_data_" + str(io) + "_" + self.h + "_" + str(io),
                    "logs_dir_IO_" + str(io) + "_" + self.h + "_" + str(io),
                    "val_csv_IO_" + str(io) + "_" + self.h + "_" + str(io) + ".csv",
                )
            )
        if model_class_str == "ConvLSTM":
            self.model_class_str = "ConvLSTM"
            self.model_class = ConvLSTM

    def run_experiments(self):
        for t, v, l, v_csv in self.list_of_folders:
            sprint(t, v, l, v_csv)
            if self.model_class_str == "ConvLSTM":
                model = self.model_class(
                    t,
                    v,
                    l,
                    shape=(2, 1, 32, 32, 1),
                    validation_csv_file=v_csv,
                )
                model.train()
            else:
                raise (Exception("Wrong model class; Probably not implemented!"))


if __name__ == "__main__":
    th = TemporalIOLength(range(2, 9, 1), "training_data_1_4_1", "ConvLSTM").run_experiments()
