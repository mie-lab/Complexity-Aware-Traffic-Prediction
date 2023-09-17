import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file

import pathlib
import datetime
import glob
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.patches import Rectangle
from tqdm import tqdm
import datetime
from smartprint import smartprint as sprint
from slugify import slugify

sys.path.append(config.DATA_FOLDER)  # location of t4c22 folder
import t4c22
from t4c22.t4c22_config import load_basedir

from haversine import haversine, Unit
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.patches import Rectangle
import t4c22
from t4c22.t4c22_config import load_basedir


class ProcessRaw:
    def __init__(self, cityname, i_o_length, prediction_horizon, grid_size):
        """
        self, cityname, i_o_length, prediction_horizon, grid_size
        """
        self.cityname = cityname.lower()
        self.i_o_length = i_o_length
        self.grid_size = grid_size
        self.prediction_horizon = prediction_horizon
        self.create_spatial_npy_from_loop_lat_lon()

        self.date_list = None
        self.compute_matrices_for_all_dates()
        # the function compute matrices also populates the date_list
        assert isinstance(self.date_list, list)

        self.create_train_val_data_points_x_y()

        self.offset = 96 - (prediction_horizon + i_o_length * 2 + 1)  # one time for ip; one for op; one for pred_horiz;
        # self.offset replaces 96 to account for edge effects of specific experiments
    def create_spatial_npy_from_loop_lat_lon(self):
        BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
        city = self.cityname

        nodes = pandas.read_parquet(BASEDIR / "road_graph" / city / "road_graph_nodes.parquet")
        # nodes = pandas.read_parquet(os.path.join(BASEDIR, "data", "road_graph", city, "road_graph_nodes.parquet"))

        counters = nodes[nodes["counter_info"] != ""]

        daily_counts = pandas.read_parquet(BASEDIR / "loop_counter" / city / "counters_daily_by_node.parquet")
        # daily_counts = pandas.read_parquet(os.path.join(BASEDIR, "data", "loop_counter", city, "counters_daily_by_node.parquet"))

        daily_counts = daily_counts.reset_index()
        daily_counts["counter_info"] = daily_counts["counter_info"].apply(lambda x: x[0])

        nodes["counter_info"] = nodes["counter_info"].astype(str)
        daily_counts["counter_info"] = daily_counts["counter_info"].astype(str)

        self.df = pandas.merge(daily_counts, counters, on=["node_id", "counter_info"])

    def compute_matrices_for_all_dates(self):
        end_date = config.DATA_END_DATE[self.cityname]
        delta = datetime.timedelta(days=1)

        dates = []

        n = self.grid_size

        start_date = config.DATA_START_DATE[self.cityname]

        # We put a higher range of 235 even though the actual number
        # of dates for a city is around 220, but some cities have
        # 1-2 days extra
        for i in tqdm(range(235), desc=self.cityname + "_" + str(self.grid_size) + "_" + "days processed: "):
            if start_date <= end_date:
                dates.append(start_date.strftime("%Y-%m-%d"))
                start_date += delta

                M = self.helper_convert_df_to_nxnx96(self.df, n, dates[-1])
            else:
                sprint(start_date, end_date)
                sprint("Reached end of dates, skipping....Total #dates processed=", len(dates))
                self.date_list = dates
                return

        # mask = np.random.rand(M.shape[0], M.shape[1]) * 0
        # mask[np.sum(M, axis=2) > 0] = 1
        # np.save("data_samples/M.npy", M)
        # plt.imshow(mask)
        # plt.show()

    def helper_convert_df_to_nxnx96(self, df, n, date_):
        """
        n: Scale; 32 implies 32X32
        """
        #     print (get_NS_EW_distance(df))
        if not os.path.exists(os.path.join(config.DATA_FOLDER, "data_samples")):
            os.mkdir(os.path.join(config.DATA_FOLDER, "data_samples"))
        npy_filename = os.path.join(
            config.DATA_FOLDER, "data_samples", self.cityname + "_" + date_ + "-" + str(n) + "x" + str(n) + "x96.npy"
        )
        if os.path.exists(npy_filename):
            return

        df = pandas.DataFrame(df[df["day"] == date_])

        try:
            assert df.shape[0] >= 10
        except AssertionError:
            sprint(df.shape[0])
            sprint(self.cityname)
            print("Missing day: ", date_)
            pass
        # return

        x = []
        y = []
        val = []
        for i in tqdm(range(df.shape[0]), desc="creating_df"):
            #         if i > 10000:
            #             continue
            x.append(df.iloc[i].x)
            y.append(df.iloc[i].y)
            val.append(df.iloc[i].volume)

        data = {"x": x, "y": y, "volume": val}

        df = pandas.DataFrame(data)

        df = df.fillna(0)

        # sanity check
        if config.DEBUG:
            self.get_NS_EW_distance(df)

        # compute the x and y bin edges
        x_bin_edges = np.linspace(df["x"].min(), df["x"].max(), n)
        y_bin_edges = np.linspace(df["y"].min(), df["y"].max(), n)

        # compute the x and y bin indices for each point
        x_bins = np.digitize(df["x"], x_bin_edges) - 1
        y_bins = np.digitize(df["y"], y_bin_edges) - 1

        # create an empty matrix M
        M = np.zeros((n, n, 96))

        # populate the matrix M with the values from the dataframe
        count_dict = {}
        for i in range(df.shape[0]):  # , desc="Creating Matrix"):
            M[x_bins[i], y_bins[i], :] += df["volume"][i]
            if (x_bins[i], y_bins[i]) in count_dict:
                count_dict[x_bins[i], y_bins[i]] += 1
            else:
                count_dict[x_bins[i], y_bins[i]] = 1

        M = np.nan_to_num(M, 0)

        # convert sum to mean
        for i in range(df.shape[0]):
            M[x_bins[i], y_bins[i], :] = M[x_bins[i], y_bins[i], :] / count_dict[x_bins[i], y_bins[i]]

        temp_file_name = str(int(np.random.rand() * 100000000000)) + ".npy"
        np.save(temp_file_name, M)
        os.rename(temp_file_name, npy_filename)

        return M

    def get_NS_EW_distance(self, df):
        """
        compute the distance between the east-most and west-most points
        returns tuple (float, float)
        """
        east_most = df["x"].max()
        west_most = df["x"].min()
        EW = haversine((east_most, 0), (west_most, 0), unit=Unit.KILOMETERS)

        # compute the distance between the north-most and south-most points
        north_most = df["y"].max()
        south_most = df["y"].min()
        NS = haversine((0, north_most), (0, south_most), unit=Unit.KILOMETERS)

        sprint(NS, EW)
        return NS, EW

    def create_train_val_data_points_x_y(self):
        """
        ih: input horizon
        ph: prediction horizon
        oh: output horizon
        """
        oh = self.i_o_length
        ph = self.prediction_horizon
        ih = oh

        fnames = glob.glob(os.path.join(os.path.join(config.DATA_FOLDER, "data_samples")) + "/*x96.npy")
        fnames = [
            x
            for x in fnames
            if self.cityname in x and ("-" + str(self.grid_size) + "x" + str(self.grid_size) + "x96") in x
        ]

        # sorting by yyyy-mm-dd ensures that validation data always comes later than test
        list.sort(fnames)

        training_dates = set(self.date_list[: config.cutoff_day_number_train])
        validation_dates = set(self.date_list[config.start_day_number_val :])

        pathlib.Path(os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())).mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())).mkdir(
            parents=True, exist_ok=True
        )
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())
        self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        #     print (fnames)
        tcount = 0
        vcount = 0
        for i in tqdm(range(len(fnames)), desc=self.training_folder):
            f = fnames[i]
            m = np.load(f)
            #         print (m.shape)

            # since f.split("data_samples")[1] would give filenames like: melbourne_2020-06-07-8x8x96.npy
            # f.split("data_samples")[1].split(self.cityname)[1][1:11]= '2020-06-07'
            if f.split("data_samples")[1].split(self.cityname)[1][1:11] in training_dates:
                for j in range(ih, 96 - (oh + ph + 1)):
                    tcount += 1
                    r = tcount

                    if os.path.exists(
                        os.path.join(self.validation_folder, self.key_dimensions() + str(r) + "_x.npy")
                    ) and os.path.exists(
                        os.path.join(self.validation_folder, self.key_dimensions() + str(r) + "_y.npy")
                    ):
                        # processed data already exists
                        return

                    x = m[:, :, j - ih : j]
                    y = m[:, :, j + ph : j + ph + oh]

                    temp_file_name = str(int(np.random.rand() * 100000000000)) + ".npy"
                    np.save(temp_file_name, x)
                    os.rename(temp_file_name, 
                              os.path.join(self.training_folder, self.key_dimensions() + str(r) + "_x.npy"))

                    temp_file_name = str(int(np.random.rand() * 100000000000)) + ".npy"
                    np.save(temp_file_name, y)
                    os.rename(temp_file_name,
                              os.path.join(self.training_folder, self.key_dimensions() + str(r) + "_y.npy"))

            elif f.split("data_samples")[1].split(self.cityname)[1][1:11] in validation_dates:
                for j in range(ih, 96 - (oh + ph + 1)):
                    vcount += 1
                    x = m[:, :, j - ih : j]
                    y = m[:, :, j + ph : j + ph + oh]

                    r = vcount

                    temp_file_name = str(int(np.random.rand() * 100000000000)) + ".npy"
                    np.save(temp_file_name, x)
                    os.rename(temp_file_name,
                              os.path.join(self.validation_folder, self.key_dimensions() + str(r) + "_x.npy"))

                    temp_file_name = str(int(np.random.rand() * 100000000000)) + ".npy"
                    np.save(temp_file_name, y)
                    os.rename(temp_file_name,
                              os.path.join(self.validation_folder, self.key_dimensions() + str(r) + "_y.npy"))

        sprint(self.key_dimensions(), tcount, vcount)
        sprint(
            len(glob.glob(self.training_folder + "/" + self.key_dimensions() + "*_x.npy")),
            len(glob.glob(self.training_folder + "/" + self.key_dimensions() + "*_y.npy")),
        )
        sprint(
            len(glob.glob(self.validation_folder + "/" + self.key_dimensions() + "*_x.npy")),
            len(glob.glob(self.validation_folder + "/" + self.key_dimensions() + "*_y.npy")),
        )

    def key_dimensions(self):  # a __repr__() for the clas
        return slugify(str([self.cityname, self.i_o_length, self.prediction_horizon, self.grid_size])) + "-"

    def _clean_intermediate_files(self):
        for filenames in [
            glob.glob(os.path.join(self.validation_folder, self.key_dimensions()) + "*.npy"),
            glob.glob(os.path.join(self.training_folder, self.key_dimensions()) + "*.npy"),
        ]:
            for file in tqdm(filenames, desc="Cleaning intermediate files"):
                os.remove(file)

    @staticmethod
    def clean_intermediate_files(cityname, io_length, pred_horiz, scale):
        for filenames in [
            glob.glob(
                os.path.join(
                    os.path.join(
                        config.VALIDATION_DATA_FOLDER, ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)
                    ),
                    ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale),
                )
                + "*.npy"
            ),
            glob.glob(
                os.path.join(
                    os.path.join(
                        config.TRAINING_DATA_FOLDER, ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)
                    ),
                    ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale),
                )
                + "*.npy"
            ),
        ]:
            for file in tqdm(filenames, desc="Cleaning intermediate files"):
                os.remove(file)

    @staticmethod  # same as key_dimensions
    def file_prefix(cityname, io_length, pred_horiz, scale):  # a __repr__() for the clas
        return slugify(str([cityname.lower(), io_length, pred_horiz, scale])) + "-"

    @staticmethod
    def split_file_prefix(str_prefix):  # a __repr__() for the clas
        assert str_prefix.count("-") == 4
        city, io, pred_horiz, scale = str_prefix.split("-")[:4]

        dict_ = {}
        dict_["city"] = city
        dict_["io_length"] = io
        dict_["pred_horiz"] = pred_horiz
        dict_["scale"] = scale

        return dict_


if __name__ == "__main__":
    # os.system("rm -rf data_samples && mkdir data_samples")
    for city in config.city_list_def:
        # scales
        for scale in config.scales_def:
            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    ProcessRaw(cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale)

        # io_lengths
        # for scale in config.scales_def:
        #     for i_o_length in config.i_o_lengths:
        #         for pred_horiz in config.pred_horiz_def:
        #             ProcessRaw(cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale)
        #
        # # pred_horiz
        # for scale in config.scales_def:
        #     for i_o_length in config.i_o_lengths_def:
        #         for pred_horiz in config.pred_horiz:
        #             ProcessRaw(cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale)
