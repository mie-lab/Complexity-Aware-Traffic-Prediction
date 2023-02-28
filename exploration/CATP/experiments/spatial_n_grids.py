import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config
from models.ConvLSTM import ConvLSTM
from smartprint import smartprint as sprint


class SpatialNGrids:
    def __init__(self, range_of_n_in_nXn_grids, base_folder_name_i_h_o, model_class_str):
        """
        :param range_of_n_in_nXn_grids: list of horizons
        :param base_folder_name_i_h_o: must follow this format training_data_8_4_8
        """
        assert model_class_str in ["ConvLSTM"]  # others not implemented yet
        self.n1 = base_folder_name_i_h_o.split("xx")[0].split("hh")[1]
        self.n2 = base_folder_name_i_h_o.split("xx")[1].split("ww")[0]

        assert self.n1 == self.n2

        i, h, o = base_folder_name_i_h_o.split("hh")[0].split("_")[2:5]

        self.list_of_folders = []
        self.range_of_n = range_of_n_in_nXn_grids
        for n in self.range_of_n:
            self.postfix = "hh" + str(n) + "xx" + str(n) + "ww"
            self.list_of_folders.append(
                (
                    "training_data_" + self.postfix + "_" + str(i) + "_" + str(h) + "_" + str(o),
                    "validation_data_" + self.postfix + "_" + str(i) + "_" + str(h) + "_" + str(o),
                    "logs_dir_" + self.postfix + "_" + str(i) + "_" + str(h) + "_" + str(o),
                    "val_csv_" + self.postfix + "_" + str(i) + "_" + str(h) + "_" + str(o),
                    n,
                )
            )
        if model_class_str == "ConvLSTM":
            self.model_class_str = "ConvLSTM"
            self.model_class = ConvLSTM

    def run_experiments(self):
        for t, v, l, v_csv, N in self.list_of_folders:
            sprint(t, v, l, v_csv, N)
            if self.model_class_str == "ConvLSTM":
                model = self.model_class(
                    t,
                    v,
                    l,
                    shape=(2, 1, N, N, 1),
                    validation_csv_file=v_csv,
                )
                model.train()
            else:
                raise (Exception("Wrong model class; Probably not implemented!"))


if __name__ == "__main__":
    th = SpatialNGrids(
        [1, 16, 32, 64, 128, 252], "training_data_1_4_1_hh32xx32ww", "ConvLSTM"
    ).run_experiments()  # 1, 16, 32, 64, 128, 252
