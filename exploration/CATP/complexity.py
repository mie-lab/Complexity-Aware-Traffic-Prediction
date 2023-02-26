import os
import sys

sys.path.append(os.path.dirname(__file__))
import config
import glob
import numpy as np
from tqdm import tqdm
from smartprint import smartprint as sprint
from scipy.spatial import minkowski_distance_p
import random
import matplotlib.pyplot as plt


class complexity:
    def __init__(self, training_data_folder, y_thresh=-1, model_predict="dummy", PM=True):
        """
        tnl: temporal_neighbour_limit (defined using DL based traffic prediction)
        model_predict = model.predict Or a dummy value if Perfect model is True
        method: ["default", "fractional"]
        """
        self.method = config.cx_method
        self.training_data_folder = training_data_folder
        self.y_thresh = y_thresh
        self.tnl = config.cx_tnl
        self.N = config.cx_N
        if not config.cx_re_compute_y_thresh:
            self.y_thresh = config.cx_y_thresh
        else:
            self.y_thresh = self.determine_y_thresh_maxvar()
        self.complexity_each_sample = self.compute_criticality_smooth(
            model_predict=model_predict, PM=PM, y_thresh=self.y_thresh
        )

    # compute CSR from Arpit et al. or data-sample specific criticality
    def compute_criticality_smooth(self, model_predict, y_thresh, PM=True):
        count_CS = []
        cache_dict = {}  # to reduce the number of file reads
        hit, miss = 0, 0
        filenames = glob.glob(self.training_data_folder + "/*.npy")
        filenames = [f for f in filenames if "_x" in f]

        range_of_fnames = list(range(self.tnl + 1, len(filenames) - self.tnl))
        random.shuffle(range_of_fnames)

        total_count = 0
        for i in range_of_fnames[: self.N]:  # , desc="computing CSR for training dataset"):
            total_count += 1

            n = str(i)
            if int(n) in cache_dict:
                x, y = cache_dict[int(n)]
                hit += 1
            else:
                x = np.load(self.training_data_folder + "/" + n + "_x" + ".npy")
                y = np.load(self.training_data_folder + "/" + n + "_y" + ".npy")
                cache_dict[int(n)] = x, y
                miss += 1

            neighbours_x = []
            neighbours_y = []

            for j in range(-self.tnl, self.tnl):
                neigh = int(n) + j
                if j == 0:
                    continue
                if neigh in cache_dict:
                    x_hat, y_hat = cache_dict[neigh]
                    hit += 1
                else:
                    x_hat = np.load(self.training_data_folder + "/" + str(neigh) + "_x" + ".npy")
                    y_hat = np.load(self.training_data_folder + "/" + str(neigh) + "_y" + ".npy")
                    cache_dict[neigh] = x_hat, y_hat
                    miss += 1

                x_hat = x_hat.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
                x_hat = np.moveaxis(x_hat, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]

                y_hat = y_hat.reshape((-1, y.shape[0], y.shape[1], y.shape[2]))
                y_hat = np.moveaxis(y_hat, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]

                neighbours_x.append(x_hat)
                neighbours_y.append(y_hat)

            if PM:
                prediction = np.vstack(tuple(neighbours_y))
            else:
                prediction = model_predict(np.vstack(tuple(neighbours_x)))

            x = x.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
            x = np.moveaxis(x, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]
            y = y.reshape((-1, y.shape[0], y.shape[1], y.shape[2]))
            y = np.moveaxis(y, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]

            a, b, c, d = prediction.shape[1], prediction.shape[2], prediction.shape[3], prediction.shape[4]
            prediction = prediction.reshape((-1, a * b * c * d))
            y = y.reshape((-1, a * b * c * d))

            dist = minkowski_distance_p(prediction, y, np.inf)

            #         sprint (prediction.shape, y.shape, x.shape, x_hat.shape, y_hat.shape)

            if self.method == "default":
                # Arpit et. al's formulation (with modifications to work for regression case)
                if np.any(dist > y_thresh):
                    count_CS.append(x)
            elif self.method == "fractional":
                count_CS.append((dist > y_thresh).sum() / dist.shape[0])

            del cache_dict[int(n) - self.tnl]  # no need to retain the files which have already been read

            if config.DEBUG:
                if np.random.rand() < 0.0005:
                    sprint(hit, miss, len(cache_dict))
                    sprint(prediction.shape, y.shape, dist.shape, len(count_CS))

        if self.method in ["fractional"]:
            return count_CS  # returns a list (one value for each data sample, total length N)
        elif self.method in ["default"]:
            return len(count_CS)  # returns a single value for data set of size N

    def determine_y_thresh_maxvar(self, model_predict="dummy", PM=True):
        std = {}
        mean = {}

        for i in tqdm(np.arange(0, config.max_dist, abs(0 - config.max_dist) / 10), desc="Determining y_thresh"):
            l = self.compute_criticality_smooth(model_predict, y_thresh=i, PM=PM)
            std[i] = np.std(l)
            mean[i] = np.mean(l)

        if config.DEBUG:
            plt.plot(list(std.keys()), list(std.values()), label="var@" + str(self.N) + "points", color="blue", alpha=1)
            plt.plot(
                list(mean.keys()), list(mean.values()), label="mean@" + str(self.N) + " points", color="red", alpha=1
            )
            plt.legend(fontsize=6)
            plt.xlabel(r"y_thresh")
            plt.ylabel(r"Complexity metric " + self.method)
            plt.show()

        y_thresh = max(std, key=std.get)
        return y_thresh

    def print_complexity_params(self):
        print("Training data folder: ", self.training_data_folder)
        print("Y threshold: ", self.y_thresh)
        print("TNL: ", self.tnl)
        print("N: ", self.N)
        print("Method: ", self.method)
        print("Complexity each sample: ", self.complexity_each_sample)


if __name__ == "__main__":
    cx_metric = complexity(training_data_folder=os.path.join(config.HOME_FOLDER, "training_data_8_4_8"))
    cx_metric.print_complexity_params()
