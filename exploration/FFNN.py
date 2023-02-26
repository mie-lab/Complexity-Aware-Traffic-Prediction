#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import numpy as np
from smartprint import smartprint as sprint



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LeakyReLU, Input
import numpy as np
from tensorflow.keras import layers, losses, optimizers, models, metrics
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# In[ ]:


import glob
import numpy as np
from tqdm import tqdm
from smartprint import smartprint as sprint
from scipy.spatial import minkowski_distance_p
import random
import matplotlib.pyplot as plt


# compute CSR
def compute_criticality_smooth(model_predict, training_data_folder, y_thresh=30, tnl=8, N=1000, \
                               case="default", PM=True):
    """
    tnl: temporal_neighbour_limit
    model_predict = model.predict
    case: ["mean", "default", "max", "ratios", "min", "fractional"]
    PM: Perfect model
    """
    assert case in ["mean", "default", "max", "ratios", "min", "fractional"]
    count_CS = []
    cache_dict = {}  # to reduce the number of file reads
    hit, miss = 0, 0
    filenames = glob.glob(training_data_folder + "/*.npy")
    filenames = [f for f in filenames if "_x" in f]

    range_of_fnames = list(range(tnl + 1, len(filenames) - tnl))
    random.shuffle(range_of_fnames)

    total_count = 0
    for i in (range_of_fnames[:N]):  # , desc="computing CSR for training dataset"):
        total_count += 1

        file = filenames[i]
        n = str(i)
        if int(n) in cache_dict:
            x, y = cache_dict[int(n)]
            hit += 1
        else:
            x = np.load(training_data_folder + "/" + n + "_x" + ".npy")
            y = np.load(training_data_folder + "/" + n + "_y" + ".npy")
            cache_dict[int(n)] = x, y
            miss += 1

        neighbours_x = []
        neighbours_y = []

        for j in range(-tnl, tnl):
            neigh = int(n) + j
            if j == 0:
                continue
            if neigh in cache_dict:
                x_hat, y_hat = cache_dict[neigh]
                hit += 1
            else:
                x_hat = np.load(training_data_folder + "/" + str(neigh) + "_x" + ".npy")
                y_hat = np.load(training_data_folder + "/" + str(neigh) + "_y" + ".npy")
                cache_dict[neigh] = x_hat, y_hat
                miss += 1

            x_hat = x_hat.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
            x_hat = np.moveaxis(x_hat, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]

            y_hat = y_hat.reshape((-1, y.shape[0], y.shape[1], y.shape[2]))
            y_hat = np.moveaxis(y_hat, [0, 1, 2, 3], [0, 2, 3, 1])[..., np.newaxis]

            neighbours_x.append(x_hat)
            neighbours_y.append(y_hat)

        if PM:
            prediction = (np.vstack(tuple(neighbours_y)))
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

        if case == "mean":
            count_CS.append(np.mean(dist))
        elif case == "default":
            if np.any(dist > y_thresh):
                count_CS.append(x)
        elif case == "fractional":
            count_CS.append((dist > y_thresh).sum() / dist.shape[0])
        elif case == "max":
            count_CS.append(np.max(dist))
        elif case == "min":
            count_CS.append(np.min(dist))
        elif case == "ratios":

            x_neighbours = (np.vstack(tuple(neighbours_x)))
            #             sprint (x.shape, x_neighbours.shape, prediction.shape)
            a, b, c, d = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            x_neighbours = x_neighbours.reshape((-1, a * b * c * d))
            x = x.reshape((-1, a * b * c * d))

            distX = minkowski_distance_p(x_neighbours, x, np.inf)
            #             from IPython.core.debugger import Pdb; Pdb().set_trace()
            ratio = dist / distX
            sprint(ratio)
            count_CS.append((ratio > 1).sum() / dist.shape[0])

        del cache_dict[int(n) - tnl]  # no need to retain the files which have already been read

        if np.random.rand() < 0.0005:
            sprint(hit, miss, len(cache_dict))
            sprint(prediction.shape, y.shape, dist.shape, len(count_CS))
    if case in ["mean", "ratios", "max", "min", "fractional"]:
        return (count_CS)
    elif case in ["default"]:
        return len(count_CS)


def determine_y_thresh_by_maximising_variance_around_mean(max_dist, N, method):
    std = {}
    mean = {}
    count = 1
    for i in tqdm(np.arange(0, max_dist, abs(0 - max_dist) / 20), desc="Finding y_thresh"):
        l = compute_criticality_smooth("dummy", "training_data_8_4_8", y_thresh=i, \
                                       tnl=8, N=N, case=method, PM=True)
        std[i] = np.std(l)
        mean[i] = np.mean(l)

        count += 1
    return (std), mean



tensorflow.keras.backend.clear_session()
x = np.random.rand(16, 8, 32, 32, 1)
inp = layers.Input(shape=(None, *x.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
# x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
# x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = tensorflow.keras.models.Model(inp, x)
model.compile(
    loss=tensorflow.keras.losses.binary_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),
)

model.summary()
import numpy as np
import tensorflow
import os
import glob
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy
from smartprint import smartprint as sprint

# Create CSVLogger callback with specified filename
from tensorflow.keras.callbacks import Callback


class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        criticality = compute_criticality_smooth(self.model.predict, "training_data_8_4_8", y_thresh=1150, \
                                                 tnl=8, N=200, case="fractional", PM=False)
        sprint(np.mean(criticality))
        logs['CSR_train_data_smooth'] = np.mean(criticality)


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
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for i in indexes:
            file_x = os.path.join(self.data_dir, '{}_x.npy'.format(i))
            file_y = os.path.join(self.data_dir, '{}_y.npy'.format(i))
            x = np.load(file_x)
            y = np.load(file_y)
            x_batch.append(x)
            y_batch.append(y)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        #         x_batch = np.swapaxes(x_batch, 0, -1)
        #         y_batch = np.swapaxes(y_batch, 0, -1)
        x_batch = np.moveaxis(x_batch, [0, 1, 2, 3], [0, 2, 3, 1])
        y_batch = np.moveaxis(y_batch, [0, 1, 2, 3], [0, 2, 3, 1])

        return (x_batch[..., np.newaxis]), (y_batch[..., np.newaxis])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def my_loss_fn(y_true, y_pred):
    global M
    squared_difference = tf.square((y_true) * M - (y_pred) * M)
    return tf.reduce_mean(squared_difference, axis=-1)


def non_zero_mape(y_true, y_pred):
    t = y_true[y_true > 10]
    p = y_pred[y_true > 10]

    squared_difference = tf.abs((t - p) / t)
    return tf.reduce_mean(squared_difference, axis=-1)


for n_depth in range(1):  # [1,2,3,4,5,6]:

    # Generate some random training data
    output_channels = 1

    # Create the model
    input_shape = (32, 32, 8)

    model.summary()

    loss_fn = "mse"
    optimizer = optimizers.Adam(1e-3)
    #     optimizer = optimizers.RMSprop()
    #     optimizer = "sgd"
    #     metrics = metrics.MeanAbsoluteError()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=non_zero_mape)

    # Train the model
    batch_size = 2
    epochs = 1
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    train_data_folder = "training_data_8_4_8"
    validation_data_folder = "validation_data_8_4_8"

    num_train = len(glob.glob(train_data_folder + "/*_x.npy"))
    num_validation = len(glob.glob(validation_data_folder + "/*_x.npy"))

    r = 0.01  # np.random.rand()
    train_gen = CustomDataGenerator(data_dir=train_data_folder, \
                                    num_samples=int(num_train * r), batch_size=batch_size, shuffle=True)
    validation_gen = CustomDataGenerator(data_dir=validation_data_folder, \
                                         num_samples=int(num_validation * r), batch_size=batch_size, shuffle=True)

    filename = str(n_depth) + '_validation_loss.csv'

    csv_logger = CSVLogger(filename)
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir="log_dir")

    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    model.fit(train_gen, validation_data=validation_gen, epochs=epochs, callbacks=[earlystop, csv_logger, \
                                                                                   tensorboard_callback, \
                                                                                   ComputeMetrics()], \
                                                                                workers=6)

# In[ ]:




# In[ ]:


# 'tensorboard', '--logdir log_dir')
# %load_ext tensorboard


# In[ ]:


# In[ ]:


# In[ ]:


from smartprint import smartprint as sprint
import matplotlib.pyplot as plt
import numpy as np

for x, y in train_gen:
    print(x.shape, y.shape)
    print(np.max(x[0]))
    break

# In[ ]:


# In[ ]:


# In[ ]:


import imshowpair

for k in range(2):
    for x, y in validation_gen:
        print(x.shape, y.shape)
        print(np.max(x[0]))
        break

    #     plt.imshow(y[k, 0, :, :, 0])
    #     plt.colorbar()

    sprint(x.shape, y.shape)

    yp = model.predict(x)
    #     sprint (y.shape)

    #     plt.imshow(y[k, 0, :, :, 0])
    #     plt.colorbar()
    imshowpair.imshowpair(y[k, 0, :, :, 0], yp[k, 1, :, :, 0])
    plt.colorbar()
    plt.show()






