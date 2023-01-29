import sklearn.metrics
import timesynth as ts
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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


def time_series_to_supervised(x: list or np.array, input_seq_len, output_seq_len, tvt_ratio=[0.5, 0.5, 0], n_blocks=7):
    """
    .. ..
    :param tvt_ratio: train-test-validation
    :param n_blocks: Number of blocks for blocking time series
    """
    matrix = sliding_window_view(np.array(x), window_shape=(input_seq_len + output_seq_len))
    Y = matrix[:, -output_seq_len:]
    X = matrix[:, :input_seq_len]

    def split_a_single_block(x, y, train_test_validation_ratio: list):
        n1 = int(train_test_validation_ratio[0] * x.shape[0])
        n2 = int(train_test_validation_ratio[1] * x.shape[0])
        return_dict = {}
        return_dict["X_train"] = x[:n1, :]
        return_dict["Y_train"] = y[:n1, :]
        return_dict["X_val"] = x[n1 : n1 + n2, :]
        return_dict["Y_val"] = y[n1 : n1 + n2, :]
        return_dict["X_test"] = y[n1 + n2 :, :]
        return_dict["Y_test"] = y[n1 + n2 :, :]
        return return_dict

    data_splits = []
    blocksize = X.shape[0] // n_blocks
    for i in range(n_blocks):
        x = X[i * blocksize : (i + 1) * blocksize, :]
        y = Y[i * blocksize : (i + 1) * blocksize, :]
        print(x.shape, y.shape)
        data_splits.append(split_a_single_block(x, y, train_test_validation_ratio=tvt_ratio))

    data_dict_combined = {}
    for params in data_splits[0].keys():
        data_dict_combined[params] = np.row_stack([data_splits[i][params] for i in range(n_blocks)])
        if DEBUG:
            for i in range(n_blocks):
                sprint(params, data_splits[i][params].shape)

            for key in data_dict_combined:
                sprint(key, data_dict_combined[key].shape)
    return data_dict_combined, data_splits


def train_and_predict_RF(data_splits, data_dict_combined, max_depth=30, n_estimators=100):
    """
    :return: mape, mse, mape_test, mse_test, model_list, fitted_model_list
    """
    fitted_model_list = []
    for i in range(len(data_splits)):
        data = data_splits[i]
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_val = data["X_val"]
        Y_val = data["Y_val"]
        X_test = data["X_test"]
        Y_test = data["Y_test"]

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, Y_train)
        fitted_model_list.append(model)

    mape = []
    mse = []
    mse_test = []
    mape_test = []
    plt.clf()
    for count, model in enumerate(fitted_model_list):
        y_pred = model.predict(data_dict_combined["X_val"])
        mape.append(sklearn.metrics.mean_absolute_percentage_error(y_pred, data_dict_combined["Y_val"]))
        mse.append(sklearn.metrics.mean_absolute_error(y_pred, data_dict_combined["Y_val"]))

        y_pred = model.predict(data_dict_combined["X_val"])
        mape_test.append(sklearn.metrics.mean_absolute_percentage_error(y_pred, data_dict_combined["Y_val"]))
        mse_test.append(sklearn.metrics.mean_absolute_error(y_pred, data_dict_combined["Y_val"]))
        if DEBUG:
            if count == 0:
                for i in tqdm(range(y_pred.shape[0]), desc="Plotting"):
                    if np.random.rand() <= 1:
                        plt.plot(range(i, i + y_pred.shape[1]), y_pred[i, :], label="pred", color="red", alpha=0.4)
                        plt.plot(
                            range(i, i + y_pred.shape[1]),
                            data_dict_combined["Y_val"][i, :],
                            label="GT",
                            color="blue",
                            alpha=0.4,
                        )
    # rr = int(np.random.rand() * (y_pred.shape[0]))
    # plt.xlim(rr, rr+24)
    plt.show()
    return mape, mse, mape_test, mse_test, fitted_model_list

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
                "np.mean(mape)",
                "np.std(mape)",
                "np.mean(mse)",
                "np.std(mse)",
                "np.mean(mape_test)",
                "np.std(mape_test)",
                "np.mean(mse_test)",
                "np.std(mse_test)",
            ]
        )
        input_seq_list = [12]
        n_estimators_list = list(range(2, 100))  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        max_depth_list = [2, 3, 4, 5, 10, 20, 30]
        for input_seq_len in tqdm(input_seq_list, desc="input_seq", position=1):
            output_seq_len = 3
            for n_estimators in tqdm(n_estimators_list, desc="n_estimators", position=2):
                for max_depth in tqdm(max_depth_list, desc="max_depth", position=3):
                    data_dict_combined, data_splits = time_series_to_supervised(
                        samples, input_seq_len=input_seq_len, output_seq_len=output_seq_len
                    )
                    mape, mse, mape_test, mse_test, model_list = train_and_predict_RF(
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
                            np.mean(mape),
                            np.std(mape),
                            np.mean(mse),
                            np.std(mse),
                            np.mean(mape_test),
                            np.std(mape_test),
                            np.mean(mse_test),
                            np.std(mse_test),
                        ]
                    )
                    f.flush()

    print("Run time: ", time.time() - starttime, " seconds")
