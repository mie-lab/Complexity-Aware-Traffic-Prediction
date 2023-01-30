import sklearn.metrics
import timesynth as ts
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from smartprint import smartprint as sprint
import time
from copy import deepcopy
import csv
from tqdm import tqdm

DEBUG = False


def generate_sine_curve_time_series(noise: bool):
    time_sampler = ts.TimeSampler(stop_time=30)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=5000, keep_percentage=50)
    sinusoid = ts.signals.Sinusoidal(frequency=0.5)

    white_noise = ts.noise.GaussianNoise(std=0.3)

    timeseries = ts.TimeSeries(sinusoid)  # , noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    if not noise:
        errors *= 0
        samples = signals

    return samples, signals, errors


def plot_signals(x: list, label: list, plot_or_scatter="plot"):
    assert plot_or_scatter in ["scatter", "plot"]
    for i, series in enumerate(x):
        if plot_or_scatter == "plot":
            plt.plot(series, label=label[i])
            plt.title("Line plot")
        elif plot_or_scatter == "scatter":
            plt.scatter(x=range(len(series)), y=series, label=label[i])
            plt.title("Scatter plot")
    if plot_or_scatter == "plot":
        plt.title("Line plot")
    elif plot_or_scatter == "scatter":
        plt.title("Scatter plot")
    plt.legend()
    plt.show()


def time_series_to_supervised(x: list or np.array, input_seq_len, output_seq_len,horizon=6, tvt_ratio=[0.5, 0.5, 0], n_blocks=7):
    """
    .. ..
    :param tvt_ratio: train-test-validation
    :param n_blocks: Number of blocks for blocking time series
    """
    matrix = sliding_window_view(np.array(x), window_shape=(input_seq_len + output_seq_len + horizon))
    assert matrix.shape[1] == (input_seq_len + output_seq_len + horizon)
    Y = matrix[:, input_seq_len+horizon:input_seq_len+horizon+output_seq_len]
    X = matrix[:, :input_seq_len]

    def split_a_single_block(x, y, train_test_validation_ratio: list):
        n1 = int(train_test_validation_ratio[0] * x.shape[0])
        n2 = int(train_test_validation_ratio[1] * x.shape[0])

        split_marker = np.array(list(range(x.shape[0]))) * 0
        return_dict = {}
        return_dict["X_train"] = x[:n1, :]
        split_marker[:n1] = 1
        return_dict["Y_train"] = y[:n1, :]
        return_dict["X_val"] = x[n1 : n1 + n2, :]
        split_marker[n1 : n1 + n2] = 2
        return_dict["Y_val"] = y[n1 : n1 + n2, :]
        return_dict["X_test"] = x[n1 + n2 :, :]
        split_marker[n1 + n2 :] = 3
        return_dict["Y_test"] = y[n1 + n2 :, :]
        return return_dict, split_marker.tolist()

    data_splits = []
    blocksize = X.shape[0] // n_blocks
    split_marker_list = []
    for i in range(n_blocks):
        x = X[i * blocksize : (i + 1) * blocksize, :]
        y = Y[i * blocksize : (i + 1) * blocksize, :]
        print(x.shape, y.shape)
        return_dict, split_marker = split_a_single_block(x, y, train_test_validation_ratio=tvt_ratio)
        data_splits.append(return_dict)
        split_marker_list = split_marker_list + split_marker

    data_dict_combined = {}
    for params in data_splits[0].keys():
        data_dict_combined[params] = np.row_stack([data_splits[i][params] for i in range(n_blocks)])
        if DEBUG:
            for i in range(n_blocks):
                sprint(params, data_splits[i][params].shape)

            for key in data_dict_combined:
                sprint(key, data_dict_combined[key].shape)
    if DEBUG:
        colors = ["bgrcmykw"[i] for i in split_marker_list]
        plt.plot(split_marker_list)
        plt.title("1: Train, 2:Val, 3:Test")
        plt.show()
    return data_dict_combined, data_splits


def train_and_predict_RF(data_splits, data_dict_combined, max_depth=30, n_estimators=100):
    """
    :return: mape, mse, mape_test, mse_test, model_list, fitted_model_list
    """
    fitted_model_list = []
    R = range(len(data_splits))
    if len(data_splits) == 1: # case when we use just one block
        R = range(7)
    for i in R:
        if len(data_splits) == 1:
            data = data_splits[0]
        else:
            data = data_splits[i]
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_val = data["X_val"]
        Y_val = data["Y_val"]
        X_test = data["X_test"]
        Y_test = data["Y_test"]

        if Y_train.ndim == 1:
            # Since the default GBM only produces univariate predictions
            Y_train = np.reshape(Y_train, (-1, 1))
            Y_test = np.reshape(Y_test, (-1, 1))
            Y_val = np.reshape(Y_val, (-1, 1))

        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, Y_train)
        fitted_model_list.append(model)

    sprint(len(fitted_model_list))

    mape_val = []
    mse_val = []
    mse_test = []
    mape_test = []
    mse_train = []
    mape_train = []
    plt.clf()
    y_pred_avg = 0
    flag = True
    for count, model in enumerate(fitted_model_list):
        y_pred = model.predict(data_dict_combined["X_train"])
        mape_train.append(sklearn.metrics.mean_absolute_percentage_error(y_pred, data_dict_combined["Y_train"]))
        mse_train.append(sklearn.metrics.mean_absolute_error(y_pred, data_dict_combined["Y_train"]))

        y_pred = model.predict(data_dict_combined["X_val"])
        mape_val.append(sklearn.metrics.mean_absolute_percentage_error(y_pred, data_dict_combined["Y_val"]))
        mse_val.append(sklearn.metrics.mean_absolute_error(y_pred, data_dict_combined["Y_val"]))

        y_pred = model.predict(data_dict_combined["X_test"])
        y_pred_avg += y_pred
        mape_test.append(sklearn.metrics.mean_absolute_percentage_error(y_pred, data_dict_combined["Y_test"]))
        mse_test.append(sklearn.metrics.mean_absolute_error(y_pred, data_dict_combined["Y_test"]))

    if DEBUG:
        y_pred = y_pred_avg / (count+1)
        for i in tqdm(range(2000, 2500), desc="plotting"):# range(y_pred.shape[0]//5), desc="Plotting"):
            if np.random.rand() <= 1:

                if flag:
                    if y_pred.ndim == 1:
                        # Since the default GBM only produces univariate predictions
                        y_pred = np.reshape(y_pred, (-1,1))
                    sprint (y_pred.shape)
                    if y_pred.shape[1] == 1:
                        plot_or_scatter = plt.scatter
                    elif y_pred.shape[1] > 1:
                        plot_or_scatter = plt.plot

                    plot_or_scatter(range(i, i + y_pred.shape[1]), y_pred[i, :], label="pred", color="red", alpha=0.4)

                    plot_or_scatter(
                        range(i, i + y_pred.shape[1]),
                        data_dict_combined["Y_test"][i, :],
                        color="blue",
                        alpha=0.4,
                        label="GT"
                    )
                    flag = False
                else:
                    plot_or_scatter(range(i, i + y_pred.shape[1]), y_pred[i, :], color="red", alpha=0.4)
                    plot_or_scatter(
                        range(i, i + y_pred.shape[1]),
                        data_dict_combined["Y_test"][i, :],
                        color="blue",
                        alpha=0.4,
                    )

        # rr = int(np.random.rand() * (y_pred.shape[0]))// 5
        # rr = 2000
        # plt.xlim(rr, rr+240)
    # break # just one model
    if DEBUG:
        plt.savefig(str(int(np.random.rand() * 100000)) + ".png", dpi=400)
        plt.legend()
        plt.show()
    return mape_val, mse_val, mape_test, mse_test, mape_train, mse_train, fitted_model_list

    # y_multirf = regr_multirf.predict(X_test)
    # y_rf = regr_multirf.predict(X_test)


def plot_random_100_predictions(data):
    list_of_indices = np.random.rand(100) * len(data["Y_test"].shape[0])
    fig, ax_array = plt.subplots(10, 10, squeeze=False)
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            axes.set_title("{},{}".format(i, j))
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.plot(data["Y_test"][list_of_indices[counter], :], color="blue", label="GT")
            axes.plot(data["Y_test_pred"][list_of_indices[counter], :], color="blue", label="Prediction")
    plt.show()


def read_real_data(sensor_id, cutoff_for_plotting=1200):
    assert sensor_id <= 962
    count = 0
    a = []
    with open("PEMS_train") as f:
        for row in f:
            a = a + eval(
                "[" + str(row.split(";")[sensor_id].split(" ")).replace("[", "").replace("]", "").replace("'", "") + "]"
            )
        plt.plot(a[:cutoff_for_plotting], alpha=0.5)
        plt.show()
    return a


if __name__ == "__main__":

    starttime = time.time()
    # samples, signals, errors = generate_sine_curve_time_series(noise=False)
    # plot_signals([samples, signals, errors], ["samples", "signals", "errors"])
    # plot_signals([samples, signals, errors], ["samples", "signals", "errors"], plot_or_scatter="scatter")

    samples = read_real_data(sensor_id=1, cutoff_for_plotting=1200)

    with open("results.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(
            [
                "input_seq_len",
                "output_seq_len",
                "n_estimators",
                "max_depth",
                "np.mean(mape_val)",
                "np.std(mape_val)",
                "np.mean(mse_val)",
                "np.std(mse_val)",
                "np.mean(mape_test)",
                "np.std(mape_test)",
                "np.mean(mse_test)",
                "np.std(mse_test)",
                "np.mean(mape_train)",
                "np.std(mape_train)",
                "np.mean(mse_train)",
                "np.std(mse_train)",
            ]
        )
        input_seq_list = [12]
        n_estimators_list = list(range(1, 300, 5))  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        max_depth_list = [20] # list(range(1, 150, 5))  # , 4, 5, 10, 20, 30]
        for input_seq_len in tqdm(input_seq_list, desc="input_seq", position=1):
            output_seq_len = 1
            for n_estimators in tqdm(n_estimators_list, desc="n_estimators", position=2):
                for max_depth in tqdm(max_depth_list, desc="max_depth", position=3):
                    data_dict_combined, data_splits = time_series_to_supervised(
                        samples,
                        input_seq_len=input_seq_len,
                        output_seq_len=output_seq_len,
                        tvt_ratio=[0.4, 0.4, 0.2],
                        n_blocks=7,
                        horizon=6
                    )
                    mape_val, mse_val, mape_test, mse_test, mape_train, mse_train, model_list = train_and_predict_RF(
                        data_splits=data_splits,
                        data_dict_combined=data_dict_combined,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                    )

                    # print(
                    #     [
                    #         input_seq_len,
                    #         output_seq_len,
                    #         n_estimators,
                    #         max_depth,
                    #         np.mean(mape),
                    #         np.std(mape),
                    #         np.mean(mse),
                    #         np.std(mse),
                    #         np.mean(mape_test),
                    #         np.std(mape_test),
                    #         np.mean(mse_test),
                    #         np.std(mse_test),
                    #     ]
                    # )
                    csvwriter.writerow(
                        [
                            input_seq_len,
                            output_seq_len,
                            n_estimators,
                            max_depth,
                            np.mean(mape_val),
                            np.std(mape_val),
                            np.mean(mse_val),
                            np.std(mse_val),
                            np.mean(mape_test),
                            np.std(mape_test),
                            np.mean(mse_test),
                            np.std(mse_test),
                            np.mean(mape_train),
                            np.std(mape_train),
                            np.mean(mse_train),
                            np.std(mse_train),
                        ]
                    )
                    f.flush()

    print("Run time: ", time.time() - starttime, " seconds")
