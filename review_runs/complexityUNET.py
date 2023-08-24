import os
import sys

# import tensorflow
from smartprint import smartprint as sprint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from preprocessing.ProcessRaw import ProcessRaw
from preprocessing.datagen import CustomDataGenerator
import numpy as np
import glob
import random
from tqdm import tqdm
import tensorflow
import matplotlib.pyplot as plt
import time


class Complexity:
    def __init__(
        self,
        cityname,
        i_o_length,
        prediction_horizon,
        grid_size,
        thresh,
        perfect_model,
        model_func,
        model_train_gen,
        run_pm,
        run_nm,
        run_gb,
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
        self.model_train_gen = model_train_gen
        self.thresh = thresh

        self.offset = 96 - (prediction_horizon + i_o_length * 2 + 1)  # one time for ip; one for op; one for pred_horiz;
        # self.offset replaces 96 to account for edge effects of specific experiments

        self.CSR_MP_no_thresh_mean = "NULL"
        self.CSR_MP_no_thresh_median = "NULL"
        self.CSR_MP_count_y_exceeding_r_x = "NULL"
        self.CSR_MP_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_MP_sum_y_exceeding_r_x_mean = "NULL"
        self.CSR_MP_sum_exp_y_exceeding_r_x_mean = "NULL"
        self.CSR_MP_no_thresh_frac_mean_2 = "NULL"
        self.CSR_MP_no_thresh_frac_mean_2_exp = "NULL"
        self.CSR_MP_no_thresh_frac_sum = "NULL"
        self.CSR_MP_red_by_grey_sum = "NULL"


        self.CSR_PM_no_thresh_mean = "NULL"
        self.CSR_PM_no_thresh_median = "NULL"
        self.CSR_PM_count_y_exceeding_r_x = "NULL"
        self.CSR_PM_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_PM_sum_y_exceeding_r_x_mean = "NULL"
        self.CSR_PM_sum_exp_y_exceeding_r_x_mean = "NULL"
        self.CSR_PM_no_thresh_frac_mean_2 = "NULL"
        self.CSR_PM_no_thresh_frac_mean_2_exp = "NULL"
        self.CSR_PM_no_thresh_frac_sum = "NULL"
        self.CSR_PM_red_by_grey_sum = "NULL"


        self.CSR_NM_no_thresh_mean = "NULL"
        self.CSR_NM_no_thresh_median = "NULL"
        self.CSR_NM_count_y_exceeding_r_x = "NULL"
        self.CSR_NM_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_NM_sum_y_exceeding_r_x_mean = "NULL"
        self.CSR_NM_sum_exp_y_exceeding_r_x_mean = "NULL"
        self.CSR_NM_no_thresh_frac_mean_2 = "NULL"
        self.CSR_NM_no_thresh_frac_mean_2_exp = "NULL"
        self.CSR_NM_no_thresh_frac_sum = "NULL"
        self.CSR_NM_red_by_grey_sum = "NULL"


        self.CSR_GB_no_thresh_mean = "NULL"
        self.CSR_GB_no_thresh_median = "NULL"
        self.CSR_GB_count_y_exceeding_r_x = "NULL"
        self.CSR_GB_sum_y_exceeding_r_x_max = "NULL"
        self.CSR_GB_sum_y_exceeding_r_x_mean = "NULL"
        self.CSR_GB_sum_exp_y_exceeding_r_x_mean = "NULL"
        self.CSR_GB_no_thresh_frac_mean_2 = "NULL"
        self.CSR_GB_no_thresh_frac_mean_2_exp = "NULL"
        self.CSR_GB_no_thresh_frac_sum = "NULL"
        self.CSR_GB_red_by_grey_sum = "NULL"

        self.CSR_PM_sum_y_exceeding_r_x_max_scales = np.random.rand(self.grid_size, self.grid_size) * 0

        if perfect_model:
            assert model_func == None
            self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)


        else:
            assert model_func != None
            self.model_predict = model_func

            self.cx_whole_dataset_m_predict(temporal_filter=True)

            if run_pm:
                self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)
            if run_nm:
                self.cx_whole_dataset_NM_no_thresh(temporal_filter=True)
            if run_gb:
                self.cx_whole_dataset_Garbage_predict(temporal_filter=True)
            # self.cx_whole_dataset_m_predict_slow(temporal_filter=True)
            self.csv_format()

    def cx_whole_dataset_PM_no_thresh(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        if config.cx_special_case_validation_data:
            self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.file_prefix)
        else:
            self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())

        # obj = ProcessRaw(
        #     cityname=self.cityname,
        #     i_o_length=self.i_o_length,
        #     prediction_horizon=self.prediction_horizon,
        #     grid_size=self.grid_size,
        # )

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []
        frac_sum_dataset = []

        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []
        red_by_grey_sum_dataset = []


        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.load((self.validation_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

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
                for day in config.cx_range_day_scan:
                    for width in config.cx_range_t_band_scan:  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                        ) or not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            continue

                        neighbour_indexes.append(index_with_offset)

            sum_x_m_predict = []
            sum_y_m_predict = []
            criticality_2 = []
            criticality_2_exp = []

            for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
                fileindices = neighbour_indexes[j : j + config.cx_batch_size]
                if 0 in fileindices:
                    print("Skipped file indexed with 0")
                    continue

                # sprint (len(self.model_train_gen.__getitem__(fileindices)))

                x_neighbour, y_neighbour = self.model_train_gen.__getitem__(fileindices)

                # Since this is the no thresh case
                # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                    j + config.cx_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                y_reshaped = y[np.newaxis, :-1, :-1, ...]
                x_reshaped = x[np.newaxis, :-1, :-1, ...]

                assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
                assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)

                dist_y = np.max((abs(y_neighbour - y_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
                dist_x = np.max((abs(x_neighbour - x_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)

                # Shape of y, y_neighbour etc.:  (2, 4, 25, 25, 1); 2 is the batch size here

                if config.DEBUG:
                    # should be same order;
                    # implies we do not need to recompute the x distances, but just to be safe
                    # we recompute the distances nevertheless, since it is super fast (i/o is the bottleneck)
                    sprint(sum_x)
                    sprint(dist_x)

                sum_x_m_predict.extend(dist_x.tolist())
                sum_y_m_predict.extend(dist_y.tolist())

                frac = dist_y / dist_x
                frac = frac[~np.isnan(frac)]

                criticality_2.extend(frac.tolist())
                criticality_2_exp.extend(np.exp(-frac).tolist())

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue
          
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)

            frac_sum_dataset.append(mean_y / mean_x)

            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)
            red_by_grey_sum_dataset.append(np.sum(sum_y_more_than_max_x)/np.sum(sum_x_m_predict))
            print ("parsing_for_temporal_criticality:", self.cityname, self.i_o_length, self.prediction_horizon,
                   self.grid_size,fileindex_orig, sum_y_more_than_max_x_dataset[-1])


            sum_y_more_than_mean_x = sum_y_m_predict[(sum_y_m_predict > mean_x)]
            if len(sum_y_more_than_mean_x.tolist()) > 0:
                sum_y_more_than_mean_x_dataset.append(np.mean(sum_y_more_than_mean_x))
            else:
                sum_y_more_than_mean_x_dataset.append(0)

            sum_y_more_than_mean_x_exp = np.exp(-np.abs(sum_y_m_predict - mean_x))
            if len(sum_y_more_than_mean_x_exp.tolist()) > 0:
                sum_y_more_than_mean_x_exp_dataset.append(np.mean(sum_y_more_than_mean_x_exp))

            sum_y_dataset.append(np.mean(sum_y_m_predict))
            sum_x_dataset.append(np.mean(sum_x_m_predict))

            if config.DEBUG:
                assert len(sum_x_dataset) == len(sum_y_dataset)
                sprint(len(sum_y))

        self.CSR_PM_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_PM_no_thresh_median = np.median(sum_y_dataset)
        self.CSR_PM_no_thresh_frac_sum = np.mean(frac_sum_dataset)

        self.CSR_PM_count_y_exceeding_r_x = np.mean(count_y_more_than_max_x_dataset)
        self.CSR_PM_red_by_grey_sum = np.sum(red_by_grey_sum_dataset)

        self.CSR_PM_sum_y_exceeding_r_x_max = np.sum(sum_y_more_than_max_x_dataset)
        self.CSR_PM_sum_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_dataset)
        self.CSR_PM_sum_exp_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_exp_dataset)

        self.CSR_PM_no_thresh_frac_mean_2 = np.mean(criticality_dataset_2)
        self.CSR_PM_no_thresh_frac_mean_2_exp = np.mean(criticality_dataset_2_exp)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_dataset, bins=np.arange(0, 5000, 5000 / 200))
            plt.savefig("plots/PM_sum_y/PM_sum_y" + str(round(time.time(), 2)) + ".png")
            # MP_more_max MP_sum_y MP_mean MP_mean_exp_ MP_frac_2_ MP_frac_2_exp_
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/PM_more_max/PM_more_max_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/PM_mean/PM_mean_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_exp_dataset, bins=np.arange(0, 1, 1 / 20))
            plt.savefig("plots/PM_mean_exp_/PM_mean_exp_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2, bins=np.arange(0, 10, 10 / 100))
            plt.ylim(0, 400)
            plt.savefig("plots/PM_frac_2_/PM_frac_2_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2_exp, bins=np.arange(0, 1, 1 / 100))
            plt.ylim(0, 200)
            plt.savefig("plots/PM_frac_2_exp_/PM_frac_2_exp_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_NM_no_thresh(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())

        # obj = ProcessRaw(
        #     cityname=self.cityname,
        #     i_o_length=self.i_o_length,
        #     prediction_horizon=self.prediction_horizon,
        #     grid_size=self.grid_size,
        # )

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []
        frac_sum_dataset = []

        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []
        red_by_grey_sum_dataset = []


        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = x

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
                for day in config.cx_range_day_scan:
                    for width in config.cx_range_t_band_scan:  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                        ) or not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            continue

                        neighbour_indexes.append(index_with_offset)

            sum_x_m_predict = []
            sum_y_m_predict = []
            criticality_2 = []
            criticality_2_exp = []

            for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
                fileindices = neighbour_indexes[j : j + config.cx_batch_size]
                if 0 in fileindices:
                    print("Skipped file indexed with 0")
                    continue

                # sprint (len(self.model_train_gen.__getitem__(fileindices)))

                x_neighbour, _ = self.model_train_gen.__getitem__(fileindices)

                y_neighbour = x_neighbour

                # Since this is the no thresh case
                # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                    j + config.cx_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                y_reshaped = y[np.newaxis, :-1, :-1, ...]
                x_reshaped = x[np.newaxis, :-1, :-1, ...]

                assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
                assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)

                dist_y = np.max((abs(y_neighbour - y_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
                dist_x = np.max((abs(x_neighbour - x_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)

                if config.DEBUG:
                    # should be same order;
                    # implies we do not need to recompute the x distances, but just to be safe
                    # we recompute the distances nevertheless, since it is super fast (i/o is the bottleneck)
                    sprint(sum_x)
                    sprint(dist_x)

                sum_x_m_predict.extend(dist_x.tolist())
                sum_y_m_predict.extend(dist_y.tolist())

                frac = dist_y / dist_x
                frac = frac[~np.isnan(frac)]

                criticality_2.extend(frac.tolist())
                criticality_2_exp.extend(np.exp(-frac).tolist())

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)

            frac_sum_dataset.append(mean_y / mean_x)

            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)
            red_by_grey_sum_dataset.append(np.sum(sum_y_more_than_max_x)/np.sum(sum_x_m_predict))

            sum_y_more_than_mean_x = sum_y_m_predict[(sum_y_m_predict > mean_x)]
            if len(sum_y_more_than_mean_x.tolist()) > 0:
                sum_y_more_than_mean_x_dataset.append(np.mean(sum_y_more_than_mean_x))
            else:
                sum_y_more_than_mean_x_dataset.append(0)

            sum_y_more_than_mean_x_exp = np.exp(-np.abs(sum_y_m_predict - mean_x))
            if len(sum_y_more_than_mean_x_exp.tolist()) > 0:
                sum_y_more_than_mean_x_exp_dataset.append(np.mean(sum_y_more_than_mean_x_exp))

            sum_y_dataset.append(np.mean(sum_y_m_predict))
            sum_x_dataset.append(np.mean(sum_x_m_predict))

            if config.DEBUG:
                assert len(sum_x_dataset) == len(sum_y_dataset)
                sprint(len(sum_y))

        self.CSR_NM_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_NM_no_thresh_median = np.median(sum_y_dataset)
        self.CSR_NM_no_thresh_frac_sum = np.mean(frac_sum_dataset)

        self.CSR_NM_count_y_exceeding_r_x = np.mean(count_y_more_than_max_x_dataset)
        self.CSR_NM_red_by_grey_sum = np.sum(red_by_grey_sum_dataset)

        self.CSR_NM_sum_y_exceeding_r_x_max = np.sum(sum_y_more_than_max_x_dataset)
        self.CSR_NM_sum_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_dataset)
        self.CSR_NM_sum_exp_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_exp_dataset)

        self.CSR_NM_no_thresh_frac_mean_2 = np.mean(criticality_dataset_2)
        self.CSR_NM_no_thresh_frac_mean_2_exp = np.mean(criticality_dataset_2_exp)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_dataset, bins=np.arange(0, 5000, 5000 / 200))
            plt.savefig("plots/NM_sum_y/NM_sum_y" + str(round(time.time(), 2)) + ".png")
            # MP_more_max MP_sum_y MP_mean MP_mean_exp_ MP_frac_2_ MP_frac_2_exp_
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/NM_more_max/NM_more_max_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/NM_mean/NM_mean_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_exp_dataset, bins=np.arange(0, 1, 1 / 20))
            plt.savefig("plots/NM_mean_exp_/NM_mean_exp_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2, bins=np.arange(0, 10, 10 / 100))
            plt.ylim(0, 400)
            plt.savefig("plots/NM_frac_2_/NM_frac_2_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2_exp, bins=np.arange(0, 1, 1 / 100))
            plt.ylim(0, 200)
            plt.savefig("plots/NM_frac_2_exp_/NM_frac_2_exp_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_Garbage_predict(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())

        # obj = ProcessRaw(
        #     cityname=self.cityname,
        #     i_o_length=self.i_o_length,
        #     prediction_horizon=self.prediction_horizon,
        #     grid_size=self.grid_size,
        # )

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []
        frac_sum_dataset = []

        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []
        red_by_grey_sum_dataset = []


        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.random.random_sample(x.shape) * np.max(x.flatten())

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
                for day in config.cx_range_day_scan:
                    for width in config.cx_range_t_band_scan:  # 1 hour before and after
                        current_offset = day * self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        # Test if x_neighbours and y_neighbours both exist;
                        if not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                        ) or not os.path.exists(
                            (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                        ):
                            count_missing += 1
                            # print ("Point ignored; x or y label not found; edge effect")
                            continue

                        neighbour_indexes.append(index_with_offset)

            sum_x_m_predict = []
            sum_y_m_predict = []
            criticality_2 = []
            criticality_2_exp = []

            for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
                fileindices = neighbour_indexes[j : j + config.cx_batch_size]
                if 0 in fileindices:
                    print("Skipped file indexed with 0")
                    continue

                # sprint (len(self.model_train_gen.__getitem__(fileindices)))

                x_neighbour, _ = self.model_train_gen.__getitem__(fileindices)

                y_neighbour = np.random.random_sample(x_neighbour.shape) * np.max(x_neighbour.flatten())

                # Since this is the no thresh case
                # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                    j + config.cx_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                y_reshaped = y[np.newaxis, :-1, :-1, ...]
                x_reshaped = x[np.newaxis, :-1, :-1, ...]

                assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
                assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)

                dist_y = np.max((abs(y_neighbour - y_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
                dist_x = np.max((abs(x_neighbour - x_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)

                if config.DEBUG:
                    # should be same order;
                    # implies we do not need to recompute the x distances, but just to be safe
                    # we recompute the distances nevertheless, since it is super fast (i/o is the bottleneck)
                    sprint(sum_x)
                    sprint(dist_x)

                sum_x_m_predict.extend(dist_x.tolist())
                sum_y_m_predict.extend(dist_y.tolist())

                frac = dist_y / dist_x
                frac = frac[~np.isnan(frac)]

                criticality_2.extend(frac.tolist())
                criticality_2_exp.extend(np.exp(-frac).tolist())

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)

            frac_sum_dataset.append(mean_y / mean_x)

            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)
            red_by_grey_sum_dataset.append(np.sum(sum_y_more_than_max_x)/np.sum(sum_x_m_predict))

            sum_y_more_than_mean_x = sum_y_m_predict[(sum_y_m_predict > mean_x)]
            if len(sum_y_more_than_mean_x.tolist()) > 0:
                sum_y_more_than_mean_x_dataset.append(np.mean(sum_y_more_than_mean_x))
            else:
                sum_y_more_than_mean_x_dataset.append(0)

            sum_y_more_than_mean_x_exp = np.exp(-np.abs(sum_y_m_predict - mean_x))
            if len(sum_y_more_than_mean_x_exp.tolist()) > 0:
                sum_y_more_than_mean_x_exp_dataset.append(np.mean(sum_y_more_than_mean_x_exp))

            sum_y_dataset.append(np.mean(sum_y_m_predict))
            sum_x_dataset.append(np.mean(sum_x_m_predict))

            if config.DEBUG:
                assert len(sum_x_dataset) == len(sum_y_dataset)
                sprint(len(sum_y))

        self.CSR_GB_no_thresh_mean = np.mean(sum_y_dataset)
        self.CSR_GB_no_thresh_median = np.median(sum_y_dataset)
        self.CSR_GB_no_thresh_frac_sum = np.mean(frac_sum_dataset)

        self.CSR_GB_count_y_exceeding_r_x = np.mean(count_y_more_than_max_x_dataset)
        self.CSR_GB_red_by_grey_sum = np.sum(red_by_grey_sum_dataset)

        self.CSR_GB_sum_y_exceeding_r_x_max = np.sum(sum_y_more_than_max_x_dataset)
        self.CSR_GB_sum_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_dataset)
        self.CSR_GB_sum_exp_y_exceeding_r_x_mean = np.mean(sum_y_more_than_mean_x_exp_dataset)

        self.CSR_GB_no_thresh_frac_mean_2 = np.mean(criticality_dataset_2)
        self.CSR_GB_no_thresh_frac_mean_2_exp = np.mean(criticality_dataset_2_exp)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_dataset, bins=np.arange(0, 5000, 5000 / 200))
            plt.savefig("plots/GB_sum_y/GB_sum_y" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/GB_more_max/GB_more_max_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/GB_mean/GB_mean_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_exp_dataset, bins=np.arange(0, 1, 1 / 20))
            plt.savefig("plots/GB_mean_exp_/GB_mean_exp_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2, bins=np.arange(0, 10, 10 / 100))
            plt.ylim(0, 400)
            plt.savefig("plots/GB_frac_2_/GB_frac_2_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2_exp, bins=np.arange(0, 1, 1 / 100))
            plt.ylim(0, 200)
            plt.savefig("plots/GB_frac_2_exp_/GB_frac_2_exp_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_m_predict(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """

        # self.model_train_gen is set from ConvLSTM class; It is traingen when computing cx; and
        # val_gen when computing errors
        # So, we need to set the foldernames accordingly

        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)


        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []
        frac_sum_dataset = []

        mse_y_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []
        red_by_grey_sum_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = self.model_predict(x[np.newaxis, :-1, :-1, :])

            neighbour_indexes = []

            for day in config.cx_range_day_scan:
                for width in config.cx_range_t_band_scan:  # 1 hour before and after
                    current_offset = day * self.offset + width

                    if current_offset == 0 or fileindex_orig + current_offset == 0:
                        # ignore the same point
                        # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                        continue
                    index_with_offset = fileindex_orig + current_offset

                    # Test if x_neighbours and y_neighbours both exist;
                    if not os.path.exists(
                        (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
                    ) or not os.path.exists(
                        (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
                    ):
                        count_missing += 1
                        # print ("Point ignored; x or y label not found; edge effect")
                        continue

                    neighbour_indexes.append(index_with_offset)

            sum_x_m_predict = []
            sum_y_m_predict = []
            criticality_2 = []
            criticality_2_exp = []

            for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size

                if config.cx_post_model_loading_from_saved_val_error_plots_spatial_save_spatial_npy:
                    fileindices = neighbour_indexes[j : j + config.cx_batch_size]
                    if 0 in fileindices:
                        print("Skipped file indexed with 0")
                        continue

                    # sprint (len(self.model_train_gen.__getitem__(fileindices)))
                    # try:
                    #     x_neighbour, _ = self.model_train_gen.__getitem__(fileindices)
                    # except FileNotFoundError:
                    #     print ("FileNotFoundError; Ignored: ")
                    #     continue

                    x_neighbour, _ = self.model_train_gen.__getitem__(fileindices)
                    y_neighbour = self.model_predict(x_neighbour)

                    # Since this is the no thresh case
                    # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                    assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                        j + config.cx_batch_size >= len(neighbour_indexes)
                    )  # for the last batch

                    assert x_neighbour.shape[0] == y_neighbour.shape[0]

                    y_reshaped = y # [np.newaxis, :-1, :-1, ...]
                    x_reshaped = x[np.newaxis, :-1, :-1, ...]

                    assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
                    assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)

                    dist_y = np.max((abs(y_neighbour - y_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
                    dist_x = np.max((abs(x_neighbour - x_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)


                sum_x_m_predict.extend(dist_x.tolist())
                sum_y_m_predict.extend(dist_y.tolist())

                frac = dist_y / dist_x
                frac = frac[~np.isnan(frac)]

                criticality_2.extend(frac.tolist())
                criticality_2_exp.extend(np.exp(-frac).tolist())


            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)

            frac_sum_dataset.append(mean_y / mean_x)

            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)

        self.CSR_MP_sum_y_exceeding_r_x_max = np.sum(sum_y_more_than_max_x_dataset)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_dataset, bins=np.arange(0, 5000, 5000 / 200))
            plt.savefig("plots/MP_sum_y/MP_sum_y" + str(round(time.time(), 2)) + ".png")
            # MP_more_max MP_sum_y MP_mean MP_mean_exp_ MP_frac_2_ MP_frac_2_exp_
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/MP_more_max/MP_more_max_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/MP_mean/MP_mean_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(sum_y_more_than_mean_x_exp_dataset, bins=np.arange(0, 1, 1 / 20))
            plt.savefig("plots/MP_mean_exp_/MP_mean_exp_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2, bins=np.arange(0, 10, 10 / 100))
            plt.ylim(0, 400)
            plt.savefig("plots/MP_frac_2_/MP_frac_2_" + str(round(time.time(), 2)) + ".png")

            plt.clf()
            plt.hist(criticality_dataset_2_exp, bins=np.arange(0, 1, 1 / 100))
            plt.ylim(0, 200)
            plt.savefig("plots/MP_frac_2_exp_/MP_frac_2_exp_" + str(round(time.time(), 2)) + ".png")

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
            self.CSR_MP_no_thresh_mean,
            self.CSR_MP_no_thresh_median,
            self.CSR_MP_count_y_exceeding_r_x,
            self.CSR_MP_sum_y_exceeding_r_x_max,
            self.CSR_MP_sum_y_exceeding_r_x_mean,
            self.CSR_MP_sum_exp_y_exceeding_r_x_mean,
            self.CSR_MP_no_thresh_frac_mean_2,
            self.CSR_MP_no_thresh_frac_mean_2_exp,
            self.CSR_MP_no_thresh_frac_sum,
            self.CSR_PM_no_thresh_mean,
            self.CSR_PM_no_thresh_median,
            self.CSR_PM_count_y_exceeding_r_x,
            self.CSR_PM_sum_y_exceeding_r_x_max,
            self.CSR_PM_sum_y_exceeding_r_x_mean,
            self.CSR_PM_sum_exp_y_exceeding_r_x_mean,
            self.CSR_PM_no_thresh_frac_mean_2,
            self.CSR_PM_no_thresh_frac_mean_2_exp,
            self.CSR_PM_no_thresh_frac_sum,
            self.CSR_NM_no_thresh_mean,
            self.CSR_NM_no_thresh_median,
            self.CSR_NM_count_y_exceeding_r_x,
            self.CSR_NM_sum_y_exceeding_r_x_max,
            self.CSR_NM_sum_y_exceeding_r_x_mean,
            self.CSR_NM_sum_exp_y_exceeding_r_x_mean,
            self.CSR_NM_no_thresh_frac_mean_2,
            self.CSR_NM_no_thresh_frac_mean_2_exp,
            self.CSR_NM_no_thresh_frac_sum,
            self.CSR_GB_no_thresh_mean,
            self.CSR_GB_no_thresh_median,
            self.CSR_GB_count_y_exceeding_r_x,
            self.CSR_GB_sum_y_exceeding_r_x_max,
            self.CSR_GB_sum_y_exceeding_r_x_mean,
            self.CSR_GB_sum_exp_y_exceeding_r_x_mean,
            self.CSR_GB_no_thresh_frac_mean_2,
            self.CSR_GB_no_thresh_frac_mean_2_exp,
            self.CSR_GB_no_thresh_frac_sum,
            self.CSR_GB_red_by_grey_sum,
            self.CSR_NM_red_by_grey_sum,
            self.CSR_PM_red_by_grey_sum,
            self.CSR_MP_red_by_grey_sum,
            sep=",",
        )
        print("###################################################")


if __name__ == "__main__":
    # combo experiments
    # for scale in config.scales:  # [25, 35, 45, 55, 65, 75, 85, 105]:
    #     for city in config.city_list_def:
    #         for i_o_length in config.i_o_lengths_def:
    #             for pred_horiz in config.pred_horiz:
    #                 for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
    #                     obj = ProcessRaw(
    #                         cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
    #                     )
    #                     train_data_folder = os.path.join(
    #                         config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, obj.key_dimensions()
    #                     )
    #                     num_train = len(
    #                         glob.glob(
    #                             os.path.join(config.HOME_FOLDER, train_data_folder)
    #                             + "/"
    #                             + obj.key_dimensions()
    #                             + "*_x.npy"
    #                         )
    #                     )
    #                     train_gen = CustomDataGenerator(
    #                         city,
    #                         i_o_length,
    #                         pred_horiz,
    #                         scale,
    #                         data_dir=train_data_folder,
    #                         num_samples=num_train,
    #                         batch_size=config.cl_batch_size,
    #                         shuffle=True,
    #                     )
    #                     cx = Complexity(
    #                         city,
    #                         i_o_length=i_o_length,
    #                         prediction_horizon=pred_horiz,
    #                         grid_size=scale,
    #                         thresh=thresh,
    #                         perfect_model=True,
    #                         model_func=None,
    #                         model_train_gen=train_gen,
    #                         run_pm=False,
    #                         run_nm=False,
    #                         run_gb=False,
    #                     )
    #                     cx.print_params()
    #                     cx.csv_format()

    ##########################################

    # io_lengths
    for scale in [85]: #config.scales:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list:
            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
                        obj = ProcessRaw(
                            cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                        )
                        if config.cx_special_case_validation_data:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, obj.key_dimensions()
                            )
                        else:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, obj.key_dimensions()
                            )
                        num_train = len(
                            glob.glob(
                                os.path.join(config.HOME_FOLDER, train_data_folder)
                                + "/"
                                + obj.key_dimensions()
                                + "*_x.npy"
                            )
                        )
                        train_gen = CustomDataGenerator(
                            city,
                            i_o_length,
                            pred_horiz,
                            scale,
                            data_dir=train_data_folder,
                            num_samples=num_train,
                            batch_size=config.cl_batch_size,
                            shuffle=True,
                        )
                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=train_gen,
                            run_pm=False,
                            run_nm=False,
                            run_gb=False,
                        )
                        cx.print_params()
                        cx.csv_format()
                        # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)

    """
    for scale in config.scales_def:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list:
            for i_o_length in config.i_o_lengths:
                for pred_horiz in config.pred_horiz_def:
                    for thresh in [100]:  # , 200, 400, 600, 800, 1100, 1300, 1500, 2000, 2500, 3000, 3500]:
                        obj = ProcessRaw(
                            cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                        )
                        if config.cx_special_case_validation_data:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, obj.key_dimensions()
                            )
                        else:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, obj.key_dimensions()
                            )
                        num_train = len(
                            glob.glob(
                                os.path.join(config.HOME_FOLDER, train_data_folder)
                                + "/"
                                + obj.key_dimensions()
                                + "*_x.npy"
                            )
                        )
                        train_gen = CustomDataGenerator(
                            city,
                            i_o_length,
                            pred_horiz,
                            scale,
                            data_dir=train_data_folder,
                            num_samples=num_train,
                            batch_size=config.cl_batch_size,
                            shuffle=True,
                        )
                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=train_gen,
                            run_pm=False,
                            run_nm=False,
                            run_gb=False,
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
                        if config.cx_special_case_validation_data:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, obj.key_dimensions()
                            )
                        else:
                            train_data_folder = os.path.join(
                                config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, obj.key_dimensions()
                            )
                        num_train = len(
                            glob.glob(
                                os.path.join(config.HOME_FOLDER, train_data_folder)
                                + "/"
                                + obj.key_dimensions()
                                + "*_x.npy"
                            )
                        )
                        train_gen = CustomDataGenerator(
                            city,
                            i_o_length,
                            pred_horiz,
                            scale,
                            data_dir=train_data_folder,
                            num_samples=num_train,
                            batch_size=config.cl_batch_size,
                            shuffle=True,
                        )
                        cx = Complexity(
                            city,
                            i_o_length=i_o_length,
                            prediction_horizon=pred_horiz,
                            grid_size=scale,
                            thresh=thresh,
                            perfect_model=True,
                            model_func=None,
                            model_train_gen=train_gen,
                            run_pm=False,
                            run_nm=False,
                            run_gb=False,
                        )
                        cx.print_params()
                        cx.csv_format()
                        # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)
    """
        # To parse the results into a csv:
        # grep 'for_parser:' complexity_PM.txt | sed 's/for_parser:,//g' | sed '1 i\cityname,i_o_length,prediction_horizon,grid_size,thresh,cx_sample_whole_data,cx_sample_single_point,CSR_PM_frac,CSR_PM_count,CSR_PM_no_thresh_median,CSR_PM_no_thresh_mean,CSR_PM_no_thresh_frac_median,CSR_PM_no_thresh_frac_mean'
# mkdir MP_more_max MP_sum_y MP_mean MP_mean_exp_ MP_frac_2_ MP_frac_2_exp_ PM_more_max PM_sum_y PM_mean PM_mean_exp_ PM_frac_2_ PM_frac_2_exp_ GB_more_max GB_sum_y GB_mean GB_mean_exp_ GB_frac_2_ GB_frac_2_exp_ NM_more_max NM_sum_y NM_mean NM_mean_exp_ NM_frac_2_ NM_frac_2_exp_
