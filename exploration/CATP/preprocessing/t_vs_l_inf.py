import os
import sys

import tensorflow
from smartprint import smartprint as sprint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from preprocessing.datagen import CustomDataGenerator
from scipy.spatial import minkowski_distance_p
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import seaborn as sns


class TemporalVsLinf:
    def __init__(self, training_data_folder, N, tnl):
        self.training_data_folder = training_data_folder
        num_files = len(glob.glob(os.path.join(config.DATA_FOLDER, self.training_data_folder) + "/*_x.npy"))
        self.N = min(N, num_files)
        self.tnl = tnl
        self.plot_T_vs_Linf()

    def plot_T_vs_Linf(self):
        train_gen = CustomDataGenerator(
            data_dir=self.training_data_folder, num_samples=self.N, batch_size=1, shuffle=True
        )
        temp_dist = []
        linf_dist = []
        color_list = []
        cm = plt.cm.get_cmap("RdYlBu")
        dict_ = {}

        for i in tqdm(range(self.N)):
            x, y, indexes = train_gen.custom_get_item_with_file_name(i)
            # sprint(indexes)
            if i + self.tnl > self.N:
                continue

            for j in range(1, self.tnl + 1):
                temp_dist.append(j)
                x_hat, y_hat, _ = train_gen.custom_get_item_with_file_name(None, i + j)
                a, b, c, d, e = x_hat.shape

                x_hat = x_hat.reshape((-1, b * c * d * e))
                x = x.reshape((-1, b * c * d * e))

                linf_dist.append((minkowski_distance_p(x_hat, x, np.inf))[0])
                # linf_dist.append(np.abs(np.mean(x_hat.flatten())-np.mean(x.flatten())))
                color_list.append(cm(j / self.tnl))

                if j in dict_:
                    dict_[j].append(linf_dist[-1])
                else:
                    dict_[j] = [(linf_dist[-1])]
                # if i == 0:
                # plt.scatter(temp_dist[-1], linf_dist[-1], color=color_list[-1], label="tnl " + str(j))

            # print("Train datagen shape:", x.shape, y.shape, indexes)
            # break
            # if i % 100 == 0:

            # plt.scatter(temp_dist, linf_dist) #, color=color_list)
            # plt.xlabel(r"$t_x-t_{\hat{x}}$")
            # plt.ylabel(r"|$x-\hat{x}\|_{L_\infty}$")
            # plt.ylim([0, 5000])
            # # plt.yscale("log")
            # plt.legend(fontsize=5)
            # # plt.colorbar(sc)
        sns.violinplot(data=pd.DataFrame(dict_))
        # fig, ax = plt.subplots()
        # ax.boxplot(dict_.values())
        # ax.set_xticklabels(dict_.keys())
        plt.xticks(rotation=90, fontsize=5)
        plt.xlabel(r"$t_x-t_{\hat{x}}$")
        plt.ylabel(r"|$x-\hat{x}\|_{L_\infty}$")
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_FOLDER, "Linf_vs_time.png"), dpi=300)
        plt.show()

        plt.xlabel(r"$\frac{\|x-\hat{x}\|_{L_\infty}}{t_x-t_{\hat{x}}}$")
        plt.ylabel("bin count")
        plt.hist(np.array(linf_dist) / np.array(temp_dist), 200)
        plt.title(r"Histogram of change in L$_infty$ vs temporal distance")
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_FOLDER, "Linf_vs_time_Hist.png"), dpi=300)
        plt.show()

        plt.xlabel(r"${\|x-\hat{x}\|_{L_\infty}}$")
        plt.ylabel("bin count")
        plt.hist(np.array(linf_dist), 200)
        plt.title(r"Histogram of L$_\infty$")
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_FOLDER, "Linf_Hist.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    TemporalVsLinf("training_data_hh32xx32ww_1_4_1", N=20000, tnl=16)
