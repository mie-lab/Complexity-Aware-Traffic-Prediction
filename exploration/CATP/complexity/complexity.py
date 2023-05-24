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

class Complexity:
    def __init__(self, cityname, i_o_length, prediction_horizon, grid_size, thresh, perfect_model, model_func, model_train_gen):
        """
        self, cityname, i_o_length, prediction_horizon, grid_size
        """
        self.cityname = cityname.lower()
        self.i_o_length = i_o_length
        self.grid_size = grid_size
        self.prediction_horizon = prediction_horizon
        self.file_prefix = ProcessRaw.file_prefix(cityname=self.cityname, io_length=self.i_o_length, \
                                                  pred_horiz=self.prediction_horizon, scale=self.grid_size)
        self.thresh = thresh

        self.offset = prediction_horizon + i_o_length * 2 # one time for ip; one for op; one for pred_horiz
        # self.offset replaces 96 to account for edge effects of specific experiments

        self.CSR_PM_frac = "NULL"
        self.CSR_PM_count = "NULL"
        self.CSR_PM_neighbour_stats = "NULL"
        self.CSR_PM_no_thresh_mean  = "NULL"
        self.CSR_PM_no_thresh_median = "NULL"
        self.CSR_PM_no_thresh_frac_mean = "NULL"
        self.CSR_PM_no_thresh_frac_median = "NULL"


        if perfect_model:
            assert model_func == None
            # self.cx_whole_dataset_PM(temporal_filter=True)
            self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)
        else:
            assert model_func != None
            self.model_predict = model_func
            self.model_train_gen = model_train_gen
            self.cx_whole_dataset_PM_no_thresh(temporal_filter=True)
            self.cx_whole_dataset_m_predict(temporal_filter=True)

    def compute_dist_N_points(file_list, query_point):
        random.shuffle(file_list)
        distances = []
        for i in (range(1, len(file_list))):
            neighbour_x_array = np.load(file_list[i])

            if "_x.npy" not in file_list[i]:
                raise Exception(
                    "Wrong file supplied; we should not have _y files\n since we are looking for n-hood of x")
            distances.append(np.max(np.abs(query_point - neighbour_x_array)))
        return distances

    def cx_whole_dataset_PM(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(cityname=self.cityname, i_o_length=self.i_o_length, \
                         prediction_horizon=self.prediction_horizon, grid_size=self.grid_size)
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]
        csr_count = 0

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            count_missing = 0


            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y =  np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []

            random.shuffle(file_list)


            if not temporal_filter:
                # uniform sampling case
                while (len(neighbour_indexes) < 50):
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
                    for width in range(-4, 5): # 1 hour before and after
                        current_offset = day*self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        if 0 <= index_with_offset < len(file_list):
                            # since sometimes the neighbours will not exist
                            # this can happen only at the ends of training and validation
                            # set
                            sample_point_x = np.load(file_list[index_with_offset])
                        else:
                            count_missing += 1
                            assert count_missing < config.cx_sample_whole_data//10
                            break

                        if np.max(np.abs(sample_point_x - x)) < self.thresh:
                            neighbour_indexes.append(index_with_offset)

                # sprint (len(neighbour_indexes))

            neighbour_indexes_count_list.append(len(neighbour_indexes))

            for fileindex in neighbour_indexes:
                #x_neighbor = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                try:
                    y_neighbour = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                except:
                    # FileNotFoundError: usually due to 0 index
                    print ("ERROR: FileNotFound ", (self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                    continue

                if np.max(np.abs(y_neighbour - y)) > self.thresh:
                    csr_count += 1
                    break

        if config.cx_delete_files_after_running:
            obj.clean_intermediate_files()

        self.CSR_PM_frac = csr_count / config.cx_sample_whole_data
        self.CSR_PM_count = csr_count
        self.CSR_PM_neighbour_stats = {"mean": round(np.mean(neighbour_indexes_count_list),2),
                                       "median": round(np.median(neighbour_indexes_count_list),2),
                                       "min": round(np.min(neighbour_indexes_count_list),2),
                                       "max": round(np.max(neighbour_indexes_count_list),2)}


    def cx_whole_dataset_PM_no_thresh(self, temporal_filter=False):
        """
        temporal_filter: If true, filtering is carried out using nearest neighbours
        """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(cityname=self.cityname, i_o_length=self.i_o_length, \
                         prediction_horizon=self.prediction_horizon, grid_size=self.grid_size)
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]
        criticality = []
        sum_y = []

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            count_missing = 0


            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])
            y =  np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex_orig) + "_y.npy")

            neighbour_indexes = []

            random.shuffle(file_list)


            if not temporal_filter:
                # uniform sampling case
                while (len(neighbour_indexes) < 50):
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
                    for width in range(-4, 5): # 1 hour before and after
                        current_offset = day*self.offset + width

                        if current_offset == 0 or fileindex_orig + current_offset == 0:
                            # ignore the same point
                            # fileindex_orig + current_offset == 0: since our file indexing starts from 1
                            continue
                        index_with_offset = fileindex_orig + current_offset

                        if 1 <= index_with_offset < len(file_list):
                            # since sometimes the neighbours will not exist
                            # this can happen only at the ends of training and validation
                            # set
                            sample_point_x = np.load(file_list[index_with_offset])
                        else:
                            count_missing += 1
                            assert count_missing < config.cx_sample_whole_data//10
                            break

                        # Since this is the no thresh case
                        # if np.max(np.abs(sample_point_x - x)) < self.thresh:
                        neighbour_indexes.append(index_with_offset)

                # sprint (len(neighbour_indexes))

            neighbour_indexes_count_list.append(len(neighbour_indexes))

            for fileindex in neighbour_indexes:

                try:
                    x_neighbor = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_x.npy")
                    y_neighbour = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                except:
                    # FileNotFoundError: usually due to 0 index
                    print ("ERROR: FileNotFound ", (self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                    continue

                # if np.max(np.abs(y_neighbour - y)) > self.thresh:
                criticality.append(np.max(np.abs(y_neighbour - y))/(np.max(np.abs(x_neighbor - x))))
                sum_y.append(np.max(np.abs(y_neighbour - y)))
                    # break

        if config.cx_delete_files_after_running:
            obj.clean_intermediate_files()

        self.CSR_PM_no_thresh_mean = np.mean(sum_y)
        self.CSR_PM_no_thresh_median = np.median(sum_y)
        self.CSR_PM_no_thresh_frac_mean = np.mean(criticality)
        self.CSR_PM_no_thresh_frac_median = np.median(criticality)


    def cx_whole_dataset_m_predict(self, temporal_filter=False):
        """
         temporal_filter: If true, filtering is carried out using nearest neighbours
         """
        self.training_folder = os.path.join(config.TRAINING_DATA_FOLDER, self.file_prefix)

        # we compute this information only using training data; no need for validation data
        # self.validation_folder = os.path.join(config.VALIDATION_DATA_FOLDER, self.key_dimensions())

        obj = ProcessRaw(cityname=self.cityname, i_o_length=self.i_o_length, \
                         prediction_horizon=self.prediction_horizon, grid_size=self.grid_size)
        prefix = self.file_prefix

        file_list = glob.glob(self.training_folder + "/" + self.file_prefix + "*_x.npy")
        random.shuffle(file_list)

        # file_list = file_list[:config.cx_sample_whole_data]
        criticality = []
        sum_y = []

        neighbour_indexes_count_list = []

        # sprint((config.cx_sample_single_point), len(file_list), \
        #        self.training_folder + "/" + self.file_prefix)

        for i in tqdm(range(config.cx_sample_whole_data), desc="Iterating through whole/subset of dataset"):
            count_missing = 0

            filename = file_list[i]
            x = np.load(filename)

            # get corresponding y
            fileindex_orig = int(file_list[i].split("_x.npy")[-2].split("-")[-1])

            # Only one line change from the cx_whole_dataset_PM function (Instead of simply reading, we need to predict)
            # The first newaxis is for batch, the last one is for channel
            y = self.model_predict(np.moveaxis(x, [0, 1, 2],[1, 2, 0])[ np.newaxis, ..., np.newaxis])

            neighbour_indexes = []

            random.shuffle(file_list)

            if not temporal_filter:
                # uniform sampling case
                while (len(neighbour_indexes) < 50):
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

                        if 1 <= index_with_offset < len(file_list):
                            # since sometimes the neighbours will not exist
                            # this can happen only at the ends of training and validation
                            # set
                            sample_point_x = np.load(file_list[index_with_offset])
                        else:
                            count_missing += 1
                            assert count_missing < 10
                            break

                        # if np.max(np.abs(sample_point_x - x)) < self.thresh:
                        neighbour_indexes.append(index_with_offset)

                # sprint (len(neighbour_indexes))

            neighbour_indexes_count_list.append(len(neighbour_indexes))

            # for fileindex in neighbour_indexes:
                # sprint (np.random.rand(), i)
                # try:
                #     x_neighbor = np.load((self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_x.npy")
                #
                #     # Only one line change from the cx_whole_dataset_PM function (Instead of simply reading, we need to predict)
                #     # The first newaxis is for batch, the last one is for channel
                #     y_neighbour = self.model_predict(np.moveaxis(x_neighbor, [0, 1, 2],[1, 2, 0])[ np.newaxis, ..., np.newaxis])
                # except:
                #     # FileNotFoundError: usually due to 0 index
                #     print("ERROR: FileNotFound ",
                #           (self.training_folder + "/" + self.file_prefix) + str(fileindex) + "_y.npy")
                #     continue

            for j in range(0, len(neighbour_indexes), config.cl_batch_size): # config.cl_batch_size
                fileindices = neighbour_indexes[j : j + config.cl_batch_size]
                if 0 in fileindices:
                    print ("Skipped file indexed with 0")
                    continue

                # sprint (len(self.model_train_gen.__getitem__(fileindices)))

                x_neighbour, y_neighbour = self.model_train_gen.__getitem__(fileindices)

                # Since this is the no thresh case
                # if np.max(np.abs(y_neighbour - y)) > self.thresh:

                assert ( config.cl_batch_size == x_neighbour.shape[0] ) or \
                       (j+config.cl_batch_size >= len(neighbour_indexes)) # for the last batch

                assert x_neighbour.shape[0] == y_neighbour.shape[0]

                numerator = np.max( (abs(y_neighbour - y)).reshape(x_neighbour.shape[0], -1), axis=1 )
                denominator = np.max( (abs(x_neighbour - x)).reshape(x_neighbour.shape[0], -1), axis=1 )
                frac = numerator / denominator

                criticality.extend( frac.flatten().tolist() )
                sum_y.extend( numerator.flatten().tolist() )
                # break

        # if config.cx_delete_files_after_running:
        #     obj.clean_intermediate_files()
        # This deleting should be handled while training the model; hence omitting from this function

        self.CSR_MP_no_thresh_mean = np.mean(sum_y)
        self.CSR_MP_no_thresh_median = np.median(sum_y)
        self.CSR_MP_no_thresh_frac_mean = np.mean(criticality)
        self.CSR_MP_no_thresh_frac_median = np.median(criticality)

    def print_params(self):
        print("###################################################")
        sprint (self.file_prefix)
        sprint (self.CSR_PM_frac)
        sprint (self.CSR_PM_count)
        sprint (self.CSR_PM_neighbour_stats)
        print ("###################################################")

    def csv_format(self):
        print("###################################################")
        print ("for_parser:", self.cityname, self.i_o_length, self.prediction_horizon, self.grid_size,\
               self.thresh, config.cx_sample_whole_data, config.cx_sample_single_point, \
               self.CSR_PM_frac, self.CSR_PM_count, self.CSR_PM_no_thresh_median, \
               self.CSR_PM_no_thresh_mean, self.CSR_PM_no_thresh_frac_median, self.CSR_PM_no_thresh_frac_mean, sep=",")
        print ("###################################################")



if __name__ == "__main__":

    for thresh in [500]:
        for city in config.city_list:

            # io_lengths
            for scale in config.scales_def:
                for i_o_length in config.i_o_lengths:
                    for pred_horiz in config.pred_horiz_def:
                        cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
                                        thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
                        cx.print_params()
                        cx.csv_format()

            # pred_horiz
            for repeat in range(1):
                for scale in config.scales_def:
                    for i_o_length in config.i_o_lengths_def:
                        for pred_horiz in config.pred_horiz:
                            cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
                                            thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
                            cx.print_params()
                            cx.csv_format()

            # # scales
            for scale in config.scales:
                for i_o_length in config.i_o_lengths_def:
                    for pred_horiz in config.pred_horiz_def:
                        cx = Complexity(city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale,
                                        thresh=thresh, perfect_model=True, model_func=None, model_train_gen=None)
                        cx.print_params()
                        cx.csv_format()

        # To parse the results into a csv:
        # grep 'for_parser:' complexity_PM.txt | sed 's/for_parser:,//g' | sed '1 i\cityname,i_o_length,prediction_horizon,grid_size,thresh,cx_sample_whole_data,cx_sample_single_point,CSR_PM_frac,CSR_PM_count,CSR_PM_no_thresh_median,CSR_PM_no_thresh_mean,CSR_PM_no_thresh_frac_median,CSR_PM_no_thresh_frac_mean'