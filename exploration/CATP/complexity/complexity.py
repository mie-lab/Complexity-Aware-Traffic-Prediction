import os
import sys

# import tensorflow
from smartprint import smartprint as sprint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from preprocessing.ProcessRaw import ProcessRaw

import numpy as np
import glob
import random
from tqdm import tqdm
import tensorflow


class Complexity:
    def __init__(
        self, cityname, i_o_length, prediction_horizon, grid_size, thresh, perfect_model, model_func, model_train_gen
    ):
        """
        self, cityname, i_o_length, prediction_horizon, grid_size
        """
        self.cityname = cityname.lower()
        self.i_o_length = i_o_length
        self.grid_size = grid_size
        self.prediction_horizon = prediction_horizon
        self.file_prefix = ProcessRaw.file_prefix(
            cityname=self.cityname, io_length=self.i_o_length, pred_horiz=self.prediction_horizon, scale=self.grid_size
        )
        self.thresh = thresh

        self.offset = 96 - (prediction_horizon + i_o_length * 2)  # one time for ip; one for op; one for pred_horiz
        # self.offset replaces 96 to account for edge effects of specific experiments

        self.CSR_PM_frac = "NULL"
        self.CSR_PM_count = "NULL"
        self.CSR_PM_no_thresh_mean = "NULL"
        self.CSR_PM_no_thresh_median = "NULL"
        self.CSR_PM_no_thresh_frac_mean = "NULL"
        self.CSR_PM_no_thresh_frac_median = "NULL"
        self.CSR_PM_no_thresh_frac_mean_exp = "NULL"
        self.CSR_PM_no_thresh_frac_mean_exp = "NULL"
        self.CSR_PM_no_thresh_frac_exp_mean = "NULL"
        self.CSR_PM_no_thresh_frac_exp_mean_signed = "NULL"
        self.CSR_PM_count_y_exceeding_r_x = "NULL"
        self.CSR_PM_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_PM_y_dist_mse = "NULL"

        self.CSR_MP_frac = "NULL"
        self.CSR_MP_count = "NULL"
        self.CSR_MP_no_thresh_mean = "NULL"
        self.CSR_MP_no_thresh_median = "NULL"
        self.CSR_MP_no_thresh_frac_mean = "NULL"
        self.CSR_MP_no_thresh_frac_median = "NULL"
        self.CSR_MP_no_thresh_frac_mean_exp = "NULL"
        self.CSR_MP_no_thresh_frac_mean_exp = "NULL"
        self.CSR_MP_no_thresh_frac_exp_mean = "NULL"
        self.CSR_MP_no_thresh_frac_exp_mean_signed = "NULL"
        self.CSR_MP_count_y_exceeding_r_x = "NULL"
        self.CSR_MP_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_MP_y_dist_mse = "NULL"

        self.CSR_NM_frac = "NULL"
        self.CSR_NM_count = "NULL"
        self.CSR_NM_no_thresh_mean = "NULL"
        self.CSR_NM_no_thresh_median = "NULL"
        self.CSR_NM_no_thresh_frac_mean = "NULL"
        self.CSR_NM_no_thresh_frac_median = "NULL"
        self.CSR_NM_no_thresh_frac_mean_exp = "NULL"
        self.CSR_NM_no_thresh_frac_mean_exp = "NULL"
        self.CSR_NM_no_thresh_frac_exp_mean = "NULL"
        self.CSR_NM_no_thresh_frac_exp_mean_signed = "NULL"
        self.CSR_NM_count_y_exceeding_r_x = "NULL"
        self.CSR_NM_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_NM_y_dist_mse = "NULL"


        if perfect_model:
            assert model_func == None
            # self.cx_whole_dataset_PM(temporal_filter=True)
            self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)
            self.cx_whole_dataset_NM_no_thresh(temporal_filter=True)

        else:
            assert model_func != None
            self.model_predict = model_func
            self.model_train_gen = model_train_gen

            self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)
            self.cx_whole_dataset_NM_no_thresh(temporal_filter=True)
            self.cx_whole_dataset_m_predict(temporal_filter=True)
            self.csv_format()

    # def compute_dist_N_points(file_list, query_point):
    #     random.shuffle(file_list)
    #     distances = []
    #     for i in (range(1, len(file_list))):
    #         neighbour_x_array = np.load(file_list[i])
    #
    #         if "_x.npy" not in file_list[i]:
    #             raise Exception(
    #                 "Wrong file supplied; we should not have _y files\n since we are looking for n-hood of x")
    #         distances.append(np.max(np.abs(query_point - neighbour_x_array)))
    #     return distances

    def cx_whole_dataset_PM_no_thresh(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(
            cityname=self.cityname,
            i_o_length=self.i_o_length,
            prediction_horizon=self.prediction_horizon,
            grid_size=self.grid_size,
        )
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]

        exp_criticality_frac = []

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        criticality_dataset = []
        criticality_dataset_exp = []
        criticality_dataset_exp_signed = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):

            criticality = []
            criticality_exp = []
            criticality_exp_signed = []
            mse_y = []

            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []

            if not temporal_filter:
                # uniform sampling case
                while len(neighbour_indexes) < 50:
                    random.shuffle(file_list)
                    for j in range(config.cx_sample_single_point):
                        sample_point_x = np.load(file_list[j])

                        if np.max(np.abs(sample_point_x - x)) < self.thresh:
                            fileindex = int(file_list[j].split("_x.npy")[-2].split("-")[-1])
                            neighbour_indexes.append(fileindex)
                    # sprint (len(neighbour_indexes))

                neighbour_indexes = neighbour_indexes[:50]

            elif temporal_filter:
                # Advanced filtering case
                # 3 days before, 3 days later, and today
                # within {width} on each side
                for day in range(-3, 4):
                    for width in range(-4, 5):  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ) or not os.path.exists(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            break

                        x_neighbour = np.load(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                        )

                        y_neighbour = np.load(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        )

                        i_d_i = np.max(np.abs(x_neighbour - x))

                        if i_d_i == 0:
                            sprint(index_with_offset, fileindex_orig, "x_neighbour-x=0; ignored")
                            continue  # set to smallest value so far

                        sum_x.append(i_d_i)

                        o_d_i = np.max(np.abs(y_neighbour - y))
                        sum_y.append(o_d_i)

                        mse_y.append(np.sum(np.abs(y_neighbour - y)))  # should be renamed to mae

                        criticality.append(abs(o_d_i - i_d_i) / (o_d_i + i_d_i))

                        criticality_exp.append(np.exp(abs(o_d_i - i_d_i) / (o_d_i + i_d_i)))

                        criticality_exp_signed.append(np.exp((o_d_i - i_d_i) / (o_d_i + i_d_i)))

                        assert len(sum_x) == len(sum_y)

            sum_x = np.array(sum_x)
            sum_y = np.array(sum_y)

            max_x = max(sum_x)
            max_y = max(sum_y)

            count_y_more_than_max_x = (sum_y > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            sum_y_more_than_max_x = np.sum(sum_y[(sum_y > max_x)])
            sum_y_more_than_max_x_dataset.append(sum_y_more_than_max_x)

            mse_y_dataset.append(np.sum(mse_y))

            criticality_dataset.append(np.mean(criticality))
            criticality_dataset_exp.append(np.mean(criticality_exp))
            criticality_dataset_exp_signed.append(np.mean(criticality_exp_signed))

            sum_y_dataset.append(np.mean(sum_y))
            sum_x_dataset.append(np.mean(sum_x))

            assert len(sum_x_dataset) == len(sum_y_dataset)
            # sprint (len(sum_y))

        # sprint (len(sum_x_dataset))

        self.CSR_PM_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_PM_no_thresh_median = np.median(sum_y_dataset)

        self.CSR_PM_no_thresh_frac_mean = np.mean(criticality_dataset)
        self.CSR_PM_no_thresh_frac_median = np.median(criticality_dataset)

        self.CSR_PM_no_thresh_frac_mean_exp = np.exp(np.mean(criticality_dataset))
        self.CSR_PM_no_thresh_frac_median_exp = np.exp(np.median(criticality_dataset))

        self.CSR_PM_no_thresh_frac_exp_mean = np.mean(criticality_dataset_exp)
        self.CSR_PM_no_thresh_frac_exp_mean_signed = np.median(criticality_dataset_exp_signed)

        self.CSR_PM_count_y_exceeding_r_x = np.sum(count_y_more_than_max_x_dataset)
        self.CSR_PM_y_dist_mse = np.sum(mse_y_dataset)

        self.CSR_PM_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)

    def cx_whole_dataset_NM_no_thresh(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(
            cityname=self.cityname,
            i_o_length=self.i_o_length,
            prediction_horizon=self.prediction_horizon,
            grid_size=self.grid_size,
        )
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]

        exp_criticality_frac = []

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        criticality_dataset = []
        criticality_dataset_exp = []
        criticality_dataset_exp_signed = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):

            criticality = []
            criticality_exp = []
            criticality_exp_signed = []
            mse_y = []

            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = x # One of two lines change compared to PM

            neighbour_indexes = []

            if not temporal_filter:
                # uniform sampling case
                while len(neighbour_indexes) < 50:
                    random.shuffle(file_list)
                    for j in range(config.cx_sample_single_point):
                        sample_point_x = np.load(file_list[j])

                        if np.max(np.abs(sample_point_x - x)) < self.thresh:
                            fileindex = int(file_list[j].split("_x.npy")[-2].split("-")[-1])
                            neighbour_indexes.append(fileindex)
                    # sprint (len(neighbour_indexes))

                neighbour_indexes = neighbour_indexes[:50]

            elif temporal_filter:
                # Advanced filtering case
                # 3 days before, 3 days later, and today
                # within {width} on each side
                for day in range(-3, 4):
                    for width in range(-4, 5):  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                                (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ) or not os.path.exists(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            break

                        x_neighbour = np.load(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                        )

                        # One of two lines change compared to PM
                        y_neighbour = x_neighbour

                        i_d_i = np.max(np.abs(x_neighbour - x))

                        if i_d_i == 0:
                            sprint(index_with_offset, fileindex_orig, "x_neighbour-x=0; ignored")
                            continue  # set to smallest value so far

                        sum_x.append(i_d_i)

                        o_d_i = np.max(np.abs(y_neighbour - y))
                        sum_y.append(o_d_i)

                        mse_y.append(np.sum(np.abs(y_neighbour - y)))  # should be renamed to mae

                        criticality.append(abs(o_d_i - i_d_i) / (o_d_i + i_d_i))

                        criticality_exp.append(np.exp(abs(o_d_i - i_d_i) / (o_d_i + i_d_i)))

                        criticality_exp_signed.append(np.exp((o_d_i - i_d_i) / (o_d_i + i_d_i)))

                        assert len(sum_x) == len(sum_y)

            sum_x = np.array(sum_x)
            sum_y = np.array(sum_y)

            max_x = max(sum_x)
            max_y = max(sum_y)

            count_y_more_than_max_x = (sum_y > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            sum_y_more_than_max_x = np.sum(sum_y[(sum_y > max_x)])
            sum_y_more_than_max_x_dataset.append(sum_y_more_than_max_x)

            mse_y_dataset.append(np.sum(mse_y))

            criticality_dataset.append(np.mean(criticality))
            criticality_dataset_exp.append(np.mean(criticality_exp))
            criticality_dataset_exp_signed.append(np.mean(criticality_exp_signed))

            sum_y_dataset.append(np.mean(sum_y))
            sum_x_dataset.append(np.mean(sum_x))

            assert len(sum_x_dataset) == len(sum_y_dataset)
            # sprint (len(sum_y))

        # sprint (len(sum_x_dataset))

        self.CSR_NM_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_NM_no_thresh_median = np.median(sum_y_dataset)

        self.CSR_NM_no_thresh_frac_mean = np.mean(criticality_dataset)
        self.CSR_NM_no_thresh_frac_median = np.median(criticality_dataset)

        self.CSR_NM_no_thresh_frac_mean_exp = np.exp(np.mean(criticality_dataset))
        self.CSR_NM_no_thresh_frac_median_exp = np.exp(np.median(criticality_dataset))

        self.CSR_NM_no_thresh_frac_exp_mean = np.mean(criticality_dataset_exp)
        self.CSR_NM_no_thresh_frac_exp_mean_signed = np.median(criticality_dataset_exp_signed)

        self.CSR_NM_count_y_exceeding_r_x = np.sum(count_y_more_than_max_x_dataset)
        self.CSR_NM_y_dist_mse = np.sum(mse_y_dataset)

        self.CSR_NM_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)



    def cx_whole_dataset_m_predict(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(
            cityname=self.cityname,
            i_o_length=self.i_o_length,
            prediction_horizon=self.prediction_horizon,
            grid_size=self.grid_size,
        )
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]

        exp_criticality_frac = []

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        criticality_dataset = []
        criticality_dataset_exp = []
        criticality_dataset_exp_signed = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):

            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []

            if not temporal_filter:
                # uniform sampling case
                while len(neighbour_indexes) < 50:
                    random.shuffle(file_list)
                    for j in range(config.cx_sample_single_point):
                        sample_point_x = np.load(file_list[j])

                        if np.max(np.abs(sample_point_x - x)) < self.thresh:
                            fileindex = int(file_list[j].split("_x.npy")[-2].split("-")[-1])
                            neighbour_indexes.append(fileindex)
                    # sprint (len(neighbour_indexes))

                neighbour_indexes = neighbour_indexes[:50]

            elif temporal_filter:
                # Advanced filtering case
                # 3 days before, 3 days later, and today
                # within {width} on each side
                for day in range(-3, 4):
                    for width in range(-4, 5):  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ) or not os.path.exists(
                            (self.training_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            break

                        neighbour_indexes.append(index_with_offset)


            sum_x_m_predict = []
            sum_y_m_predict = []
            mse_y_m_predict = []
            criticality = []

            for j in range(0, len(neighbour_indexes), config.cl_batch_size):  # config.cl_batch_size

                fileindices = neighbour_indexes[j : j + config.cl_batch_size]
                if 0 in fileindices:
                    print("Skipped file indexed with 0")
                    continue

                # sprint (len(self.model_train_gen.__getitem__(fileindices)))

                x_neighbour, y_neighbour = self.model_train_gen.__getitem__(fileindices)

                # Since this is the no thresh case
                # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                assert (config.cl_batch_size == x_neighbour.shape[0]) or (
                    j + config.cl_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                dist_y = np.max((abs(y_neighbour - y)).reshape(x_neighbour.shape[0], -1), axis=1)
                dist_x = np.max((abs(x_neighbour - x)).reshape(x_neighbour.shape[0], -1), axis=1)

                if config.DEBUG:
                    # should be same order;
                    # implies we do not need to recompute the x distances, but just to be safe
                    # we recompute the distances nevertheless, since it is super fast (i/o is the bottleneck)
                    sprint(sum_x)
                    sprint(dist_x)

                sum_x_m_predict.extend(dist_x.tolist())
                sum_y_m_predict.extend(dist_y.tolist())

                criticality.extend((np.abs(dist_y-dist_x)/np.abs(dist_y+dist_x)).tolist())

                mse_y_m_predict.extend(
                    np.sum((abs(y_neighbour - y)).reshape(x_neighbour.shape[0], -1), axis=1).tolist()
                )

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            max_x = max(sum_x_m_predict)

            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset.append(np.mean(criticality))

            sum_y_more_than_max_x = np.sum(sum_y_m_predict[(sum_y_m_predict > max_x)])
            sum_y_more_than_max_x_dataset.append(sum_y_more_than_max_x)

            mse_y_dataset.append(np.sum(mse_y_m_predict))

            sum_y_dataset.append(np.mean(sum_y_m_predict))
            sum_x_dataset.append(np.mean(sum_x_m_predict))

            if config.DEBUG:
                assert len(sum_x_dataset) == len(sum_y_dataset)
                sprint (len(sum_y))

        self.CSR_MP_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_MP_no_thresh_median = np.median(sum_y_dataset)

        self.CSR_MP_count_y_exceeding_r_x = np.sum(count_y_more_than_max_x_dataset)
        self.CSR_MP_y_dist_mse = np.sum(mse_y_dataset)

        self.CSR_MP_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)

        self.CSR_MP_no_thresh_frac_mean = np.mean(criticality_dataset)

    def print_params(self):
        supress_outputs = True
        # print("###################################################")
        # sprint(self.file_prefix)
        # sprint(self.CSR_PM_frac)
        # sprint(self.CSR_PM_count)
        # sprint(self.CSR_PM_neighbour_stats)
        # print("###################################################")

    def csv_format(self):
        print("###################################################")
        print(
            "for_parser:",
            self.cityname,
            self.i_o_length,
            self.prediction_horizon,
            self.grid_size,
            self.thresh,
            config.cx_sample_whole_data,
            config.cx_sample_single_point,
            self.CSR_PM_frac,
            self.CSR_PM_count,
            self.CSR_PM_no_thresh_mean,
            self.CSR_PM_no_thresh_median,
            self.CSR_PM_no_thresh_frac_mean,
            self.CSR_PM_no_thresh_frac_median,
            self.CSR_PM_no_thresh_frac_mean_exp,
            self.CSR_PM_no_thresh_frac_mean_exp,
            self.CSR_PM_no_thresh_frac_exp_mean,
            self.CSR_PM_no_thresh_frac_exp_mean_signed,
            self.CSR_PM_count_y_exceeding_r_x,
            self.CSR_PM_sum_y_exceeding_r_x_max,
            self.CSR_PM_y_dist_mse,
            self.CSR_MP_frac,
            self.CSR_MP_count,
            self.CSR_MP_no_thresh_mean,
            self.CSR_MP_no_thresh_median,
            self.CSR_MP_no_thresh_frac_mean,
            self.CSR_MP_no_thresh_frac_median,
            self.CSR_MP_no_thresh_frac_mean_exp,
            self.CSR_MP_no_thresh_frac_mean_exp,
            self.CSR_MP_no_thresh_frac_exp_mean,
            self.CSR_MP_no_thresh_frac_exp_mean_signed,
            self.CSR_MP_count_y_exceeding_r_x,
            self.CSR_MP_sum_y_exceeding_r_x_max,
            self.CSR_MP_y_dist_mse,
            self.CSR_NM_frac,
            self.CSR_NM_count,
            self.CSR_NM_no_thresh_mean,
            self.CSR_NM_no_thresh_median,
            self.CSR_NM_no_thresh_frac_mean,
            self.CSR_NM_no_thresh_frac_median,
            self.CSR_NM_no_thresh_frac_mean_exp,
            self.CSR_NM_no_thresh_frac_mean_exp,
            self.CSR_NM_no_thresh_frac_exp_mean,
            self.CSR_NM_no_thresh_frac_exp_mean_signed,
            self.CSR_NM_count_y_exceeding_r_x,
            self.CSR_NM_sum_y_exceeding_r_x_max,
            self.CSR_NM_y_dist_mse,
            sep=",",
        )
        print("###################################################")


if __name__ == "__main__":

    # io_lengths
    for scale in config.scales:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list:
            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
                        obj = ProcessRaw(
                            cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                        )

                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=None,
                        )
                        cx.print_params()
                        cx.csv_format()
                        # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)

    for scale in config.scales_def:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list:
            for i_o_length in config.i_o_lengths:
                for pred_horiz in config.pred_horiz_def:
                    for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
                        obj = ProcessRaw(
                            cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                        )

                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=None,
                        )
                        cx.print_params()
                        cx.csv_format()
                        # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)

    for scale in config.scales_def:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list:
            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz:
                    for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
                        obj = ProcessRaw(
                            cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                        )

                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=None,
                        )
                        cx.print_params()
                        cx.csv_format()
                        # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)

        # for scale in [55]:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        #     for i_o_length in [4]:
        #         for pred_horiz in [1, 2, 3, 4, 5, 6, 7, 8]:
        #             for thresh in [100, 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
        #                 cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                                 thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #                 cx.print_params()
        #                 cx.csv_format()
        #             obj = ProcessRaw(cityname=city, i_o_length=i_o_length, \
        #                              prediction_horizon=pred_horiz, grid_size=scale)
        #             obj.clean_intermediate_files()
        #
        #
        # for scale in [55]:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        #     for i_o_length in [1,2,3,4,5,6,7,8]:
        #         for pred_horiz in [1]:
        #             for thresh in [100, 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
        #                 cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                                 thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #                 cx.print_params()
        #                 cx.csv_format()
        #             obj = ProcessRaw(cityname=city, i_o_length=i_o_length, \
        #                              prediction_horizon=pred_horiz, grid_size=scale)
        #             obj.clean_intermediate_files()

        # for scale in config.scales:
        #     for i_o_length in config.i_o_lengths_def:
        #         for pred_horiz in config.pred_horiz_def:
        #             for thresh in [100, 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
        #                 cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                                 thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #                 cx.print_params()
        #                 cx.csv_format()
        #             obj = ProcessRaw(cityname=city, i_o_length=i_o_length, \
        #                              prediction_horizon=pred_horiz, grid_size=scale)
        #             obj.clean_intermediate_files()
        #
        # for scale in config.scales_def:
        #     for i_o_length in config.i_o_lengths_def:
        #         for pred_horiz in config.pred_horiz:
        #             for thresh in [100, 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
        #                 cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                                 thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #                 cx.print_params()
        #                 cx.csv_format()
        #             obj = ProcessRaw(cityname=city, i_o_length=i_o_length, \
        #                              prediction_horizon=pred_horiz, grid_size=scale)
        #             obj.clean_intermediate_files()

        # # pred_horiz
        # for repeat in range(1):
        #     for scale in config.scales_def:
        #         for i_o_length in config.i_o_lengths_def:
        #             for pred_horiz in config.pred_horiz:
        #                 cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                                 thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #                 cx.print_params()
        #                 cx.csv_format()
        #
        # # # scales
        # for scale in config.scales:
        #     for i_o_length in config.i_o_lengths_def:
        #         for pred_horiz in config.pred_horiz_def:
        #             cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
        #                             thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
        #             cx.print_params()
        #             cx.csv_format()

        # To parse the results into a csv:
        # grep 'for_parser:' complexity_PM.txt | sed 's/for_parser:,//g' | sed '1 i\cityname,i_o_length,prediction_horizon,grid_size,thresh,cx_sample_whole_data,cx_sample_single_point,CSR_PM_frac,CSR_PM_count,CSR_PM_no_thresh_median,CSR_PM_no_thresh_mean,CSR_PM_no_thresh_frac_median,CSR_PM_no_thresh_frac_mean'
