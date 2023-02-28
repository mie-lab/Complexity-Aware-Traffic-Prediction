import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

import pandas as pd
from smartprint import smartprint as sprint
import matplotlib.pyplot as plt


class PlotValidation:
    def __init__(self, fname_list, keylist):
        self.val_csv_file_list = fname_list
        self.dataframes = {}
        self.keylist = keylist

        for f, key in zip(self.val_csv_file_list, self.keylist):
            print(f, key)
            self.dataframes[key] = pd.read_csv(os.path.join(config.RESULTS_FOLDER, f))

        self.plot(self.dataframes, x="epoch")

    def plot(self, df_dict, x="epoch"):
        print(df_dict.keys())
        counter = 0

        for key in df_dict:
            counter += 1
            df = df_dict[key]

            df = df.rename(columns={"CSR_train_data_fractionalTrue": "PM", "CSR_train_data_fractionalFalse": "EfMC"})
            df = df.rename(columns={"CSR_train_data_PM_True": "PM", "CSR_train_data_PM_False": "EfMC"})
            # plt.clf()
            cols = df.columns.to_list()
            cols.remove("val_loss")  # removing the columns whose values are higher by several
            # orders of magnitude for ease of plotting
            cols.remove("loss")
            cols.remove("epoch")
            cols.remove("non_zero_mape")
            cols.remove("val_non_zero_mape")
            cols.remove("CSR_y_thresh_PM_False")
            cols.remove("CSR_y_thresh_PM_True")

            for c, col in enumerate(cols):
                alpha = 1 - counter / (len(df_dict)) + 0.125
                if counter == 1:
                    plt.plot(df[x].to_list(), df[col].to_list(), color="rgbyo"[c], label=col, alpha=alpha)
                else:
                    plt.plot(df[x].to_list(), df[col].to_list(), color="rgbyo"[c], alpha=alpha)

        plt.legend()
        # plt.yscale("log")
        # plt.savefig(os.path.join(config.RESULTS_FOLDER, key) + ".png")
        # plt.savefig(os.path.join(config.RESULTS_FOLDER, "combined_plot_IO") + ".png")
        # plt.savefig(os.path.join(config.RESULTS_FOLDER, "combined_plot_horiz") + ".png")
        plt.savefig(os.path.join(config.RESULTS_FOLDER, "combined_plot_grid") + ".png")


if __name__ == "__main__":
    # list_of_folders = []
    # key_list = []
    # for io in range(1, 9):
    #     list_of_folders.append("val_csv_IO_" + str(io) + "_" + "4" + "_" + str(io) + ".csv")
    #     key_list.append("IO_" + str(io))
    # PlotValidation(fname_list=list_of_folders, keylist=key_list)

    # list_of_folders = []
    # key_list = []
    # for horiz in range(1, 7):
    #     list_of_folders.append("val_csv_horiz_" + "1" + "_" + str(horiz) + "_" + "1" + ".csv")
    #     key_list.append("_horiz_" + str(horiz))
    # PlotValidation(fname_list=list_of_folders, keylist=key_list)

    list_of_folders = []
    key_list = []
    for n in [1, 16, 32, 64, 128]:  # 256]:
        list_of_folders.append("val_csv_hh" + str(n) + "xx" + str(n) + "ww_1_4_1")
        key_list.append("_N_grid_" + str(n))
    PlotValidation(fname_list=list_of_folders, keylist=key_list)
