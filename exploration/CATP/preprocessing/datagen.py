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


class CustomDataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, data_dir, num_samples, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(1, num_samples + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for i in indexes:
            file_x = os.path.join(config.HOME_FOLDER, self.data_dir, "{}_x.npy".format(i))
            file_y = os.path.join(config.HOME_FOLDER, self.data_dir, "{}_y.npy".format(i))
            x = np.load(file_x)
            y = np.load(file_y)
            x_batch.append(x)
            y_batch.append(y)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        if config.dg_debug:
            sprint(x_batch.shape, y_batch.shape)
            sprint(file_y)
            sprint(file_x)
        x_batch = np.moveaxis(x_batch, [0, 1, 2, 3], [0, 2, 3, 1])
        y_batch = np.moveaxis(y_batch, [0, 1, 2, 3], [0, 2, 3, 1])

        return (x_batch[..., np.newaxis]), (y_batch[..., np.newaxis])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    train_data_folder = "training_data_8_4_8"
    validation_data_folder = "validation_data_8_4_8"

    num_train = len(glob.glob(os.path.join(config.HOME_FOLDER, train_data_folder) + "/*_x.npy"))
    num_validation = len(glob.glob(os.path.join(config.HOME_FOLDER, validation_data_folder) + "/*_x.npy"))
    sprint(num_train, num_validation)

    r = 0.01  # np.random.rand()
    train_gen = CustomDataGenerator(
        data_dir=train_data_folder, num_samples=int(num_train * r), batch_size=21, shuffle=True
    )
    validation_gen = CustomDataGenerator(
        data_dir=validation_data_folder, num_samples=int(num_validation * r), batch_size=21, shuffle=True
    )

    for x, y in train_gen:
        print("Train datagen shape:", x.shape, y.shape)
        break

    for x, y in validation_gen:
        print("Validation datagen shape:", x.shape, y.shape)
        break
