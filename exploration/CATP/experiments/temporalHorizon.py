import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config
from models.ConvLSTM import ConvLSTM
from smartprint import smartprint as sprint


class TemporalHorizon:
    def __init__(self, range_of_horizons, base_folder_name_i_h_o, model_class_str):
        """
        :param range_of_horizons: list of horizons
        :param base_folder_name_i_h_o: must follow this format training_data_8_4_8
        """
        assert model_class_str in ["ConvLSTM"]  # others not implemented yet
        self.range_of_horizons = range_of_horizons
        _, _, self.i, _, self.o = base_folder_name_i_h_o.split("_")
        self.list_of_folders = []
        for r in self.range_of_horizons:
            self.list_of_folders.append(
                (
                    "training_data_" + self.i + "_" + str(r) + "_" + self.o,
                    "validation_data_" + self.i + "_" + str(r) + "_" + self.o,
                    "logs_dir_" + self.i + "_" + str(r) + "_" + self.o,
                    "val_csv_" + self.i + "_" + str(r) + "_" + self.o + ".csv",
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
    th = TemporalHorizon(range(1, 8, 2), "training_data_1_4_1", "ConvLSTM").run_experiments()
