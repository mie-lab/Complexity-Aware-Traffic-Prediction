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
        perfect_model,
        model_func,
        model_train_gen,
        model_predict_gen,
        run_pm,
        run_nm,
        run_gb,
        predictions_dir,
    ):
        """
        self, cityname, i_o_length, prediction_horizon, grid_size
        """
        self.predictions_dir = predictions_dir
        self.cityname = cityname.lower()
        self.i_o_length = i_o_length
        self.grid_size = grid_size
        self.prediction_horizon = prediction_horizon
        if config.cx_range_t_band_scan != -1:
            self.cx_range_t_band_scan = config.cx_range_t_band_scan
        else:
            self.cx_range_t_band_scan = range(- prediction_horizon, prediction_horizon + 1)

        self.file_prefix = ProcessRaw.file_prefix(
            cityname=self.cityname, io_length=self.i_o_length, pred_horiz=self.prediction_horizon, scale=self.grid_size
        )
        self.model_train_gen = model_train_gen
        self.model_predict_gen = model_predict_gen

        self.offset = 96 - (prediction_horizon + i_o_length * 2 + 1)  # one time for ip; one for op; one for pred_horiz;
        # self.offset replaces 96 to account for edge effects of specific experiments

        self.CSR_MP_sum_y_exceeding_r_x_max = -1
        self.CSR_PM_sum_y_exceeding_r_x_max = -1
        self.CSR_NM_sum_y_exceeding_r_x_max = 0
        self.CSR_GB_sum_y_exceeding_r_x_max = 9999
        self.CSR_MP_std = -1
        self.CSR_PM_std = -1
        
        self.CSR_MP_count, self.CSR_PM_count, self.CSR_NM_count, self.CSR_GB_count = -1, -1, -1, -1


        self.CSR_PM_sum_y_exceeding_r_x_max_scales = np.random.rand(self.grid_size, self.grid_size) * 0

        if perfect_model:
            assert model_func == None
            self.cx_whole_dataset_PM_no_thresh()
            if config.cx_spatial_cx_PM_dist_enabled:
                self.cx_whole_dataset_PM_no_thresh_spatial()

        else:
            assert model_func != None
            self.model_predict = model_func

            self.cx_whole_dataset_m_predict()

            if run_pm:
                self.cx_whole_dataset_PM_no_thresh()
            if run_nm:
                self.cx_whole_dataset_NM_no_thresh()
            if run_gb:
                self.cx_whole_dataset_Garbage_predict()
            self.csv_format()

    def cx_whole_dataset_PM_no_thresh(self):
        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(len(file_list)), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.load((self.validation_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []


            # Advanced filtering case
            # 3 days before, 3 days later, and today
            # within {width} on each side
            for day in config.cx_range_day_scan:
                for width in self.cx_range_t_band_scan:  # 1 hour before and after
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

                y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
                x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]

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


            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue

            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)


            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)
            # print ("parsing_for_temporal_criticality:", self.cityname, self.i_o_length, self.prediction_horizon,
            #        self.grid_size,fileindex_orig, sum_y_more_than_max_x_dataset[-1])


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


        self.CSR_PM_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)
        self.CSR_PM_count = len(sum_y_more_than_max_x_dataset)

        self.CSR_PM_std = np.std(sum_y_more_than_max_x_dataset)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/PM_more_max/PM_more_max_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_m_predict(self):
        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)


        file_list = glob.glob(self.predictions_dir + "/" + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(len(file_list)), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]

            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.load(self.predictions_dir + "/"  + self.file_prefix + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []

            # Advanced filtering case
            # 3 days before, 3 days later, and today
            # within {width} on each side
            for day in config.cx_range_day_scan:
                for width in self.cx_range_t_band_scan:  # 1 hour before and after
                    current_offset = day * self.offset + width

                    if current_offset == 0 or fileindex_orig + current_offset == 0:
                        # ignore the same point
                        # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                        continue
                    index_with_offset = fileindex_orig + current_offset

                    # Test if x_neighbours and y_neighbours both exist;
                    if not os.path.exists(
                            self.predictions_dir + "/" + self.file_prefix + str(index_with_offset) + "_x.npy"
                    ) or not os.path.exists(
                            self.predictions_dir + "/" + self.file_prefix + str(index_with_offset) + "_y.npy"
                    ) :
                        count_missing += 1
                        # print ("Point ignored; x or y label not found; edge effect")
                        continue

                    neighbour_indexes.append(index_with_offset)

            sum_x_m_predict = []
            sum_y_m_predict = []
            criticality_2 = []
            criticality_2_exp = []

            for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
                fileindices = neighbour_indexes[j: j + config.cx_batch_size]
                if 0 in fileindices:
                    print("Skipped file indexed with 0")
                    continue

                x_neighbour, y_neighbour = self.model_predict_gen.__getitem__(fileindices)

                assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                        j + config.cx_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
                x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]

                assert (y_reshaped.shape[1:] == y_neighbour.shape[
                                                1:])  # ignore the batch size dimension (the first one)
                assert (x_reshaped.shape[1:] == x_neighbour.shape[
                                                1:])  # ignore the batch size dimension (the first one)

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


            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue

            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)


            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)
            # print("parsing_for_temporal_criticality:", self.cityname, self.i_o_length, self.prediction_horizon,
            #       self.grid_size, fileindex_orig, sum_y_more_than_max_x_dataset[-1])

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

        self.CSR_MP_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)
        self.CSR_MP_count = len(sum_y_more_than_max_x_dataset)

        self.CSR_MP_std = np.std(sum_y_more_than_max_x_dataset)


        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/MP_more_max/MP_more_max_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_NM_no_thresh(self):
        return # No need since it is always = 0

        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []

        random.shuffle(file_list)

        for i in tqdm(range(len(file_list)), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = x

            neighbour_indexes = []



            # Advanced filtering case
            # 3 days before, 3 days later, and today
            # within {width} on each side
            for day in config.cx_range_day_scan:
                for width in self.cx_range_t_band_scan:  # 1 hour before and after
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

                y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
                x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]

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

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            # print("Length: ", len(sum_x_m_predict.tolist()))
            if len(sum_x_m_predict.tolist()) == 0:
                continue
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)


            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)

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

        self.CSR_NM_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)
        self.CSR_NM_count = len(sum_y_more_than_max_x_dataset)


        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/NM_more_max/NM_more_max_" + str(round(time.time(), 2)) + ".png")

    def cx_whole_dataset_Garbage_predict(self):
        return # Since it is no longer relevant for our RQ, output is just a high number
        # as expected
        self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())

        file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        criticality_dataset_2 = []
        criticality_dataset_2_exp = []
        count_y_more_than_max_x_dataset = []
        sum_y_more_than_max_x_dataset = []
        sum_y_more_than_mean_x_dataset = []
        sum_y_more_than_mean_x_exp_dataset = []

        sum_y_dataset = []
        sum_x_dataset = []


        random.shuffle(file_list)

        for i in tqdm(range(len(file_list)), desc="Iterating through whole/subset of dataset"):
            sum_y = []
            sum_x = []

            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y = np.random.random_sample(x.shape) * np.max(x.flatten())

            neighbour_indexes = []

            # Advanced filtering case
            # 3 days before, 3 days later, and today
            # within {width} on each side
            for day in config.cx_range_day_scan:
                    for width in self.cx_range_t_band_scan:  # 1 hour before and after
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

                x_neighbour, _ = self.model_train_gen.__getitem__(fileindices)

                y_neighbour = np.random.random_sample(x_neighbour.shape) * np.max(x_neighbour.flatten())

                assert (config.cx_batch_size == x_neighbour.shape[0]) or (
                    j + config.cx_batch_size >= len(neighbour_indexes)
                )  # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
                x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]

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

            sum_x_m_predict = np.array(sum_x_m_predict)
            sum_y_m_predict = np.array(sum_y_m_predict)

            if len(sum_x_m_predict.tolist()) == 0:
                continue
            max_x = np.max(sum_x_m_predict)
            mean_x = np.mean(sum_x_m_predict)
            mean_y = np.mean(sum_y_m_predict)


            count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
            count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)

            criticality_dataset_2.append(np.mean(criticality_2))
            criticality_dataset_2_exp.append(np.mean(criticality_2_exp))

            sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
            if len(sum_y_more_than_max_x.tolist()) > 0:
                sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
            else:
                sum_y_more_than_max_x_dataset.append(0)

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

        self.CSR_GB_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)
        self.CSR_GB_count = len(sum_y_more_than_max_x_dataset)

        if config.DEBUG:
            plt.clf()
            plt.hist(sum_y_more_than_max_x_dataset, bins=100)
            plt.xlim(0, 3000)
            plt.savefig("plots/GB_more_max/GB_more_max_" + str(round(time.time(), 2)) + ".png")

    def csv_format(self):
        print("###################################################")
        print(
            "for_parser:",
            self.cityname,
            self.i_o_length,
            self.prediction_horizon,
            self.grid_size,
            config.cx_sample_whole_data,
            self.CSR_MP_sum_y_exceeding_r_x_max,
            self.CSR_PM_sum_y_exceeding_r_x_max,
            self.CSR_NM_sum_y_exceeding_r_x_max,
            self.CSR_GB_sum_y_exceeding_r_x_max,
            self.CSR_PM_std,
            self.CSR_MP_std,
            self.CSR_MP_count, 
            self.CSR_PM_count, 
            self.CSR_NM_count, 
            self.CSR_GB_count,
            sep=",",
        )
        print("###################################################")

# def cx_whole_dataset_PM_no_thresh_spatial(self):
#     self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)
#
#     # we compute this information only using training data; no need for validation data
#     # self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.key_dimensions())
#
#     file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
#     random.shuffle(file_list)
#
#     sum_y_more_than_max_x_dataset = {}
#     for m_x in range(self.grid_size):
#         for m_y in range(self.grid_size):
#             sum_y_more_than_max_x_dataset[m_x, m_y] = []
#             sum_y_more_than_max_x_dataset[m_x, m_y] = []
#
#     random.shuffle(file_list)
#
#     for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
#         sum_y = {}
#         sum_x = {}
#
#         count_missing = 0
#
#         filename = file_list[i]
#         x = np.load(filename)
#
#         # get corresponding y
#         fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
#         y = np.load((self.validation_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")
#
#         neighbour_indexes = []
#
#
#         # Advanced filtering case
#         # 3 days before, 3 days later, and today
#         # within {width} on each side
#         for day in config.cx_range_day_scan:
#             for width in self.cx_range_t_band_scan:  # 1 hour before and after
#                 current_offset = day * self.offset + width
#
#                 if current_offset == 0 or fileindex_orig + current_offset == 0:
#                     # ignore the same point
#                     # fileindex_orig + current_offset == 0: since our file indexing starts from 1
#                     continue
#                 index_with_offset = fileindex_orig + current_offset
#
#                 # Test if x_neighbours and y_neighbours both exist;
#                 if not os.path.exists(
#                         (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
#                 ) or not os.path.exists(
#                     (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
#                 ):
#                     count_missing += 1
#                     # print ("Point ignored; x or y label not found; edge effect")
#                     continue
#
#                 neighbour_indexes.append(index_with_offset)
#
#         sum_x_m_predict = {}
#         sum_y_m_predict = {}
#         sum_y_more_than_max_x = {}
#         max_x = {}
#
#         for m_x in range(self.grid_size):
#             for m_y in range(self.grid_size):
#                 sum_x_m_predict[m_x, m_y] = []
#                 sum_y_m_predict[m_x, m_y] = []
#
#
#         for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
#             fileindices = neighbour_indexes[j: j + config.cx_batch_size]
#             if 0 in fileindices:
#                 print("Skipped file indexed with 0")
#                 continue
#
#             # sprint (len(self.model_train_gen.__getitem__(fileindices)))
#
#             x_neighbour, y_neighbour = self.model_train_gen.__getitem__(fileindices)
#
#             # Since this is the no thresh case
#             # if np.max(np.abs(y_neighbour - y)) > self.thresh:
#
#             assert (config.cx_batch_size == x_neighbour.shape[0]) or (
#                     j + config.cx_batch_size >= len(neighbour_indexes)
#             )  # for the last batch
#
#             assert x_neighbour.shape[0] == y_neighbour.shape[0]
#
#             y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
#             x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
#
#             assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
#             assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
#
#             if config.DEBUG:
#                 sprint (x_reshaped.shape, x.shape)
#                 sprint(y_reshaped.shape, y.shape)
#                 sprint(x_neighbour.shape, x_reshaped.shape, y_reshaped.shape)
#
#             for m_x in range(self.grid_size):
#                 for m_y in range(self.grid_size):
#
#                     dist_y = np.max((abs(y_neighbour[:, :, m_x:m_x+1, m_y:m_y+1, :] -
#                                          y_reshaped[:, :, m_x:m_x+1, m_y:m_y+1, :]))
#                                     .reshape(x_neighbour.shape[0], -1), axis=1)
#                     dist_x = np.max((abs(x_neighbour[:, :, m_x:m_x+1, m_y:m_y+1, :] -
#                                          x_reshaped[:, :, m_x:m_x+1, m_y:m_y+1, :]))
#                                     .reshape(x_neighbour.shape[0], -1), axis=1)
#
#                     # Shape of y, y_neighbour etc.:  (2, 4, 25, 25, 1); 2 is the batch size here
#
#                     sum_x_m_predict[m_x, m_y].extend(dist_x.tolist())
#                     sum_y_m_predict[m_x, m_y].extend(dist_y.tolist())
#
#         for m_x in range(self.grid_size):
#             for m_y in range(self.grid_size):
#                 sum_x_m_predict[m_x, m_y] = np.array(sum_x_m_predict[m_x, m_y])
#                 sum_y_m_predict[m_x, m_y] = np.array(sum_y_m_predict[m_x, m_y])
#
#                 max_x[m_x, m_y] = np.max(sum_x_m_predict[m_x, m_y])
#
#                 sum_y_more_than_max_x[m_x, m_y] = sum_y_m_predict[m_x, m_y][(sum_y_m_predict[m_x, m_y] > max_x[m_x, m_y])]
#
#                 if len(sum_y_more_than_max_x[m_x, m_y].tolist()) > 0:
#                     sum_y_more_than_max_x_dataset[m_x, m_y].append(np.sum(sum_y_more_than_max_x[m_x, m_y]))
#                 else:
#                     sum_y_more_than_max_x_dataset[m_x, m_y].append(0)
#
#     for m_x in range(self.grid_size):
#         for m_y in range(self.grid_size):
#             self.CSR_PM_sum_y_exceeding_r_x_max_scales[m_x, m_y] = np.sum(sum_y_more_than_max_x_dataset[m_x, m_y])
#
#     if not os.path.exists(os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix)):
#         os.mkdir(os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix))
#
#     np.save(
#         os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix, "_PM_spatial_complexity" +
#                      str(int(np.random.rand() * 10000000000))+ ".npy"),
#         self.CSR_PM_sum_y_exceeding_r_x_max_scales)

# def cx_whole_dataset_m_predict_legacy(self, temporal_filter=False):
#     """
#     temporal_filter: If true, filtering is carried out using nearest neighbours
#     """
#
#     # self.model_train_gen is set from ConvLSTM class; It is traingen when computing cx; and
#     # val_gen when computing errors
#     # So, we need to set the foldernames accordingly
#
#     self.validation_folder = os.path.join(config.TRAINING_DATA_FOLDER, ProcessRaw.file_prefix(cityname=self.cityname, io_length=self.i_o_length, pred_horiz=self.prediction_horizon, scale=self.grid_size))
#
#     file_list = glob.glob(self.validation_folder + "/" + self.file_prefix + "*_x.npy")
#     random.shuffle(file_list)
#
#     criticality_dataset_2 = []
#     criticality_dataset_2_exp = []
#     count_y_more_than_max_x_dataset = []
#     sum_y_more_than_max_x_dataset = []
#     sum_y_more_than_mean_x_dataset = []
#     sum_y_more_than_mean_x_exp_dataset = []
#     frac_sum_dataset = []
#
#
#     sum_y_dataset = []
#     sum_x_dataset = []
#     red_by_grey_sum_dataset = []
#
#     random.shuffle(file_list)
#
#     for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
#         sum_y = []
#         sum_x = []
#
#         count_missing = 0
#
#         filename = file_list[i]
#         x = np.load(filename)
#
#         # get corresponding y
#         fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
#         y = self.model_predict(np.moveaxis(x, [0, 1, 2], [1, 2, 0])[np.newaxis, ..., np.newaxis])
#         y = np.moveaxis(y[0, :, :, :, 0], [0, 1, 2], [2, 0, 1])
#
#         neighbour_indexes = []
#
#         if not temporal_filter:
#             # uniform sampling case
#             while len(neighbour_indexes) < 50:
#                 random.shuffle(file_list)
#                 for j in range(config.cx_sample_single_point):
#                     sample_point_x = np.load(file_list[j])
#
#                     if np.max(np.abs(sample_point_x - x)) < self.thresh:
#                         fileindex = int(file_list[j].split("_x.npy")[-2].split("-")[-1])
#                         neighbour_indexes.append(fileindex)
#                 # sprint (len(neighbour_indexes))
#
#             neighbour_indexes = neighbour_indexes[:50]
#
#         elif temporal_filter:
#             # Advanced filtering case
#             # 3 days before, 3 days later, and today
#             # within {width} on each side
#             for day in config.cx_range_day_scan:
#                 for width in self.cx_range_t_band_scan:  # 1 hour before and after
#                     current_offset = day * self.offset + width
#
#                     if current_offset == 0 or fileindex_orig + current_offset == 0:
#                         # ignore the same point
#                         # fileindex_orig + current_offset == 0: since our file indexing starts from 1
#                         continue
#                     index_with_offset = fileindex_orig + current_offset
#
#                     # Test if x_neighbours and y_neighbours both exist;
#                     if not os.path.exists(
#                         (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_x.npy"
#                     ) or not os.path.exists(
#                         (self.validation_folder + "/" + self.file_prefix) + str(index_with_offset) + "_y.npy"
#                     ):
#                         count_missing += 1
#                         # print ("Point ignored; x or y label not found; edge effect")
#                         continue
#
#                     neighbour_indexes.append(index_with_offset)
#
#         sum_x_m_predict = []
#         sum_y_m_predict = []
#         criticality_2 = []
#         criticality_2_exp = []
#
#         for j in range(0, len(neighbour_indexes), config.cx_batch_size):  # config.cl_batch_size
#
#             if config.cx_post_model_loading_from_saved_val_error_plots_spatial_save_spatial_npy:
#                 fileindices = neighbour_indexes[j : j + config.cx_batch_size]
#                 if 0 in fileindices:
#                     print("Skipped file indexed with 0")
#                     continue
#
#                 # sprint (len(self.model_train_gen.__getitem__(fileindices)))
#
#                 x_neighbour, y_neighbour_gt = self.model_train_gen.__getitem__(fileindices)
#
#                 y_neighbour = self.model_predict(x_neighbour)
#
#                 # Since this is the no thresh case
#                 # if np.max(np.abs(y_neighbour - y)) > self.thresh:
#
#                 assert (config.cx_batch_size == x_neighbour.shape[0]) or (
#                     j + config.cx_batch_size >= len(neighbour_indexes)
#                 )  # for the last batch
#
#                 assert x_neighbour.shape[0] == y_neighbour.shape[0]
#
#                 y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
#                 x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
#
#                 assert (y_reshaped.shape[1:] == y_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
#                 assert (x_reshaped.shape[1:] == x_neighbour.shape[1:]) # ignore the batch size dimension (the first one)
#
#                 dist_y = np.max((abs(y_neighbour - y_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
#                 dist_x = np.max((abs(x_neighbour - x_reshaped)).reshape(x_neighbour.shape[0], -1), axis=1)
#
#                 if not os.path.exists(os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix + "-spatial-errors")):
#                     os.mkdir(os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix + "-spatial-errors"))
#                 np.save(os.path.join(config.INTERMEDIATE_FOLDER, self.file_prefix + "-spatial-errors", str(int(np.random.rand()*10000000000)) + ".npy"),
#                         np.mean((y_neighbour - y_neighbour_gt) ** 2, axis=0))
#
#             else:
#                 # create two dummy lists dist_x and dist_y for the computation below;
#                 # this case is useful when we just want the temporal errors
#                 dist_x = np.array([1])
#                 dist_y = np.array([1])
#
#             if config.DEBUG:
#                 # should be same order;
#                 # implies we do not need to recompute the x distances, but just to be safe
#                 # we recompute the distances nevertheless, since it is super fast (i/o is the bottleneck)
#                 sprint(sum_x)
#                 sprint(dist_x)
#
#             sum_x_m_predict.extend(dist_x.tolist())
#             sum_y_m_predict.extend(dist_y.tolist())
#
#             frac = dist_y / dist_x
#             frac = frac[~np.isnan(frac)]
#
#             criticality_2.extend(frac.tolist())
#             criticality_2_exp.extend(np.exp(-frac).tolist())
#
#         # Speedup for when running only the temporal case, no spatial;
#         # Later we can run them together; but for now, they must be run separately (spatial and temporal)
#         if config.cx_post_model_loading_from_saved_val_error_plots_temporal:
#             assert  not  config.cx_post_model_loading_from_saved_val_error_plots_spatial_save_spatial_npy
#
#             # need to do this explicitly, since for computing complexiy, we don't keep the corresponding GT for this specific case
#             # we don't keep this in the model predict function; This will be present in the PM function
#             x_orig, y_gt = self.model_train_gen.__getitem__([fileindex_orig])
#
#             x_reshaped = np.moveaxis(x, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis]
#             y_reshaped = np.moveaxis(y, (0, 1, 2), (1, 2, 0))[np.newaxis, ..., np.newaxis] # remember this y is infact f(x)
#
#             assert (x_orig == x_reshaped).all()
#             assert (y_reshaped.shape == y_gt.shape)
#             print("parsing_model_predict_for_temporal_errors:", self.cityname, self.i_o_length,
#                   self.prediction_horizon,
#                   self.grid_size, fileindex_orig, np.mean((y_reshaped-y_gt) ** 2), np.mean((y_reshaped-x_reshaped) ** 2) ** 0.5)
#
#         sum_x_m_predict = np.array(sum_x_m_predict)
#         sum_y_m_predict = np.array(sum_y_m_predict)
#
#         # print("Length: ", len(sum_x_m_predict.tolist()))
#         if len(sum_x_m_predict.tolist()) == 0:
#             continue
#         max_x = np.max(sum_x_m_predict)
#         mean_x = np.mean(sum_x_m_predict)
#         mean_y = np.mean(sum_y_m_predict)
#
#         frac_sum_dataset.append(mean_y / mean_x)
#
#         count_y_more_than_max_x = (sum_y_m_predict > max_x).sum()
#         count_y_more_than_max_x_dataset.append(count_y_more_than_max_x)
#
#         criticality_dataset_2.append(np.mean(criticality_2))
#         criticality_dataset_2_exp.append(np.mean(criticality_2_exp))
#
#         sum_y_more_than_max_x = sum_y_m_predict[(sum_y_m_predict > max_x)]
#         if len(sum_y_more_than_max_x.tolist()) > 0:
#             sum_y_more_than_max_x_dataset.append(np.sum(sum_y_more_than_max_x))
#         else:
#             sum_y_more_than_max_x_dataset.append(0)
#
#
#         red_by_grey_sum_dataset.append(np.sum(sum_y_more_than_max_x)/np.sum(sum_x_m_predict))
#
#         sum_y_more_than_mean_x = sum_y_m_predict[(sum_y_m_predict > mean_x)]
#         if len(sum_y_more_than_mean_x.tolist()) > 0:
#             sum_y_more_than_mean_x_dataset.append(np.mean(sum_y_more_than_mean_x))
#         else:
#             sum_y_more_than_mean_x_dataset.append(0)
#
#         sum_y_more_than_mean_x_exp = np.exp(-np.abs(sum_y_m_predict - mean_x))
#         if len(sum_y_more_than_mean_x_exp.tolist()) > 0:
#             sum_y_more_than_mean_x_exp_dataset.append(np.mean(sum_y_more_than_mean_x_exp))
#
#         sum_y_dataset.append(np.mean(sum_y_m_predict))
#         sum_x_dataset.append(np.mean(sum_x_m_predict))
#
#         if config.DEBUG:
#             assert len(sum_x_dataset) == len(sum_y_dataset)
#             sprint(len(sum_y))
#
#
#
#     self.CSR_MP_sum_y_exceeding_r_x_max = np.mean(sum_y_more_than_max_x_dataset)
#
#
#     if config.DEBUG:
#         plt.clf()
#         plt.hist(sum_y_more_than_max_x_dataset, bins=100)
#         plt.xlim(0, 3000)
#         plt.savefig("plots/MP_more_max/MP_more_max_" + str(round(time.time(), 2)) + ".png")

if __name__ == "__main__":
    # io_lengths
    for scale in config.scales_def:  # [25, 35, 45, 55, 65, 75, 85, 105]:
        for city in config.city_list_def:
            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    obj = ProcessRaw(
                        cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                    )
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

                    predict_gen = CustomDataGenerator(
                        city,
                        i_o_length,
                        pred_horiz,
                        scale,
                        data_dir=os.path.join(config.HOME_FOLDER, "predictions_folder", obj.key_dimensions()),
                        num_samples=num_train,
                        batch_size=config.cl_batch_size,
                        shuffle=True,
                    )

                    cx = Complexity(
                        city,
                        i_o_length=i_o_length,
                        prediction_horizon=pred_horiz,
                        grid_size=scale,
                        perfect_model=True,
                        model_func=None,
                        model_train_gen=train_gen,
                        model_predict_gen=predict_gen,
                        run_pm=True,
                        run_nm=False,
                        run_gb=False,
                        predictions_dir=os.path.join(config.HOME_FOLDER, "predictions_folder", obj.key_dimensions())
                    )

                    cx.csv_format()

                    print ("Now computing cx using M_predict")
                    cx = Complexity(
                        city,
                        i_o_length=i_o_length,
                        prediction_horizon=pred_horiz,
                        grid_size=scale,
                        perfect_model=False,
                        model_func="Some model",
                        model_train_gen=train_gen,
                        model_predict_gen=predict_gen,
                        run_pm=True,
                        run_nm=False,
                        run_gb=False,
                        predictions_dir=os.path.join(config.HOME_FOLDER, "predictions_folder", obj.key_dimensions())
                    )
                    # ProcessRaw.clean_intermediate_files(city, i_o_length, pred_horiz, scale)

        # To parse the results into a csv:
        # grep 'for_parser:' complexity_PM.txt | sed 's/for_parser:,//g' | sed '1 i\cityname,i_o_length,prediction_horizon,grid_size,thresh,cx_sample_whole_data,cx_sample_single_point,CSR_PM_frac,CSR_PM_count,CSR_PM_no_thresh_median,CSR_PM_no_thresh_mean,CSR_PM_no_thresh_frac_median,CSR_PM_no_thresh_frac_mean'
# mkdir MP_more_max MP_sum_y MP_mean MP_mean_exp_ MP_frac_2_ MP_frac_2_exp_ PM_more_max PM_sum_y PM_mean PM_mean_exp_ PM_frac_2_ PM_frac_2_exp_ GB_more_max GB_sum_y GB_mean GB_mean_exp_ GB_frac_2_ GB_frac_2_exp_ NM_more_max NM_sum_y NM_mean NM_mean_exp_ NM_frac_2_ NM_frac_2_exp_
