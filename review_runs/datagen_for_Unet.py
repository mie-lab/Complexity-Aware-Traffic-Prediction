import os
import sys

import tensorflow
from smartprint import smartprint as sprint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

# In[ ]:


import glob
import numpy as np
from smartprint import smartprint as sprint
from preprocessing.ProcessRaw import ProcessRaw


class CustomDataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, cityname, io_length, pred_horiz, scale, data_dir, num_samples, batch_size=32, shuffle=True):
        self.city_name, self.io_length, self.pred_horiz, self.scale = cityname, io_length, pred_horiz, scale
        self.prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)
        self.data_dir = data_dir

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(1, num_samples + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, index):
        if isinstance(index, int):
            indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        elif isinstance(index, list):
            indexes = index

        x_batch = []
        y_batch = []

        for i in indexes:
            file_x = os.path.join(config.DATA_FOLDER, self.data_dir, self.prefix + "{}_x.npy".format(i))
            file_y = os.path.join(config.DATA_FOLDER, self.data_dir, self.prefix + "{}_y.npy".format(i))
            x = np.load(file_x)
            y = np.load(file_y)
            x_batch.append(x)
            y_batch.append(y)

        if config.dg_debug:
            sprint(len(indexes))
            sprint(self.prefix)
            sprint(x_batch[0].shape, y_batch[0].shape)
            if config.dg_debug_each_data_sample:
                for index, x in enumerate(x_batch):
                    sprint(x.shape, y.shape, index)
            sprint(file_y)
            sprint(file_x)

        x_batch = np.array(x_batch, dtype=float)
        y_batch = np.array(y_batch, dtype=float)

        if config.dg_debug:
            sprint(x_batch[0].shape, y_batch[0].shape)

        assert len(indexes) > 0

        # Only change compared to datagen_for_Unet
        # x_batch = np.moveaxis(x_batch, [0, 1, 2, 3], [0, 2, 3, 1])
        # y_batch = np.moveaxis(y_batch, [0, 1, 2, 3], [0, 2, 3, 1])

        # sprint ((x_batch[..., np.newaxis]).shape, (y_batch[..., np.newaxis]).shape)

        # Only change compared to datagen_for_Unet
        # return x_batch[:, :, :-1, :-1, :], y_batch[:, :, :-1, :-1, :]
        return x_batch[:, :-1, :-1, :], y_batch[:, :-1, :-1, :]

    # def custom_get_item_with_file_name(self, index, specific_files=None):
    #
    #     # Two functionalities:
    #     # 1. If specific files is None; this function does the same thing as __get_item__ but also returns the file
    #     #                             names.
    #     # 2. If specific files is a file number, this function returns the batch starting from the same file number
    #     # This function is used to speed up the predict for all data points, so that we can speed up the CSR computation
    #     # for each epoch
    #
    #     if specific_files == None:
    #         indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    #     else:
    #         assert isinstance(specific_files, int)
    #         indexes = [specific_files]
    #     x_batch = []
    #     y_batch = []
    #     for i in indexes:
    #         file_x = os.path.join(config.DATA_FOLDER, self.data_dir, self.prefix + "{}_x.npy".format(i))
    #         file_y = os.path.join(config.DATA_FOLDER, self.data_dir, self.prefix + "{}_y.npy".format(i))
    #         x = np.load(file_x)
    #         y = np.load(file_y)
    #         x_batch.append(x)
    #         y_batch.append(y)
    #     x_batch = np.array(x_batch)
    #     y_batch = np.array(y_batch)
    #     if config.dg_debug:
    #         sprint(x_batch.shape, y_batch.shape)
    #         sprint(file_y)
    #         sprint(file_x)
    #     x_batch = np.moveaxis(x_batch, [0, 1, 2, 3], [0, 2, 3, 1])
    #     y_batch = np.moveaxis(y_batch, [0, 1, 2, 3], [0, 2, 3, 1])
    #
    #     return (x_batch[..., np.newaxis]), (y_batch[..., np.newaxis]), indexes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    # train_data_folder = "training_data_1_4_1"
    train_data_folder = config.TRAINING_DATA_FOLDER
    # validation_data_folder = "validation_data_1_4_1"
    validation_data_folder = config.VALIDATION_DATA_FOLDER

    r = 0.06  # ratio: what fraction of data points to use

    cityname = "london"
    io_length = 4
    pred_horiz = 1
    scale = 85

    prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)
    num_train = len(glob.glob(os.path.join(config.DATA_FOLDER, train_data_folder, prefix) + "/" + prefix + "*_x.npy"))
    num_validation = len(
        glob.glob(os.path.join(config.DATA_FOLDER, validation_data_folder, prefix) + "/" + prefix + "*_x.npy")
    )
    sprint(num_train, num_validation)

    train_gen = CustomDataGenerator(
        cityname,
        io_length,
        pred_horiz,
        scale,
        data_dir=os.path.join(config.DATA_FOLDER, train_data_folder, prefix),
        num_samples=int(num_train * r),
        batch_size=config.cl_batch_size,
        shuffle=True,
    )
    validation_gen = CustomDataGenerator(
        cityname,
        io_length,
        pred_horiz,
        scale,
        data_dir=os.path.join(config.DATA_FOLDER, validation_data_folder, prefix),
        num_samples=int(num_validation * r),
        batch_size=config.cl_batch_size,
        shuffle=True,
    )

    for x, y in train_gen:
        print("Train datagen shape:", x.shape, y.shape)
        break

    for x, y in validation_gen:
        print("Validation datagen shape:", x.shape, y.shape)
        break