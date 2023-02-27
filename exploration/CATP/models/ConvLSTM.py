import os
import sys

from validation_metrics.custom_losses import non_zero_mape

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from preprocessing.datagen import CustomDataGenerator
from complexity import complexity
from smartprint import smartprint as sprint

# In[ ]:


import numpy as np
import tensorflow
import os
import glob

# Create CSVLogger callback with specified filename
from tensorflow.keras.callbacks import Callback


class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        for PM in [True, False]:
            cx = complexity(training_data_folder=self.model.training_folder, model_predict=self.model.predict, PM=PM)
            logs["CSR_train_data_" + config.cx_method + str(PM)] = np.mean(cx.complexity_each_sample)


class ConvLSTM:
    def __init__(self, training_data_folder, validation_data_folder, log_dir, shape, validation_csv_file):
        """
        Input and output shapes are the same (tuple of length 5)
        """
        self.train_data_folder = os.path.join(config.DATA_FOLDER, training_data_folder)
        self.validation_data_folder = os.path.join(config.DATA_FOLDER, validation_data_folder)
        self.shape = shape
        self.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, validation_csv_file)
        sprint (validation_csv_file, self.validation_csv_file)
        self.log_dir = os.path.join(config.INTERMEDIATE_FOLDER, log_dir)
        self.model = self.create_model()

    def create_model(self):
        _, a, b, c, d = self.shape
        x = np.random.rand(2, a, b, c, d)
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
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        # Next, we will build the complete model and compile it.
        model = tensorflow.keras.models.Model(inp, x)
        model.compile(
            loss=tensorflow.keras.losses.binary_crossentropy,
            optimizer=tensorflow.keras.optimizers.Adam(),
        )

        return model

    def train(self):
        # Train the model
        batch_size = config.cl_batch_size
        epochs = config.cl_epochs
        loss_fn = config.cl_loss_func
        optimizer = optimizers.Adam(1e-3)

        # manual reset the model since sometimes it does not reset a new model (not sure why)
        tensorflow.keras.backend.clear_session()
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=non_zero_mape)

        num_train = len(glob.glob(os.path.join(config.HOME_FOLDER, self.train_data_folder) + "/*_x.npy"))
        num_validation = len(glob.glob(os.path.join(config.HOME_FOLDER, self.validation_data_folder) + "/*_x.npy"))

        r = config.cl_percentage_of_train_data  # np.random.rand()

        train_gen = CustomDataGenerator(
            data_dir=self.train_data_folder, num_samples=int(num_train * r), batch_size=batch_size, shuffle=True
        )
        validation_gen = CustomDataGenerator(
            data_dir=self.validation_data_folder,
            num_samples=int(num_validation * r),
            batch_size=batch_size,
            shuffle=True,
        )

        csv_logger = CSVLogger(self.validation_csv_file)
        sprint (csv_logger, self.validation_data_folder)
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=self.log_dir)

        earlystop = EarlyStopping(
            monitor="val_loss", patience=config.cl_early_stopping_patience, verbose=1, mode="auto"
        )
        self.model.training_folder = self.train_data_folder

        self.model.fit(
            train_gen,
            validation_data=validation_gen,
            epochs=epochs,
            callbacks=[earlystop, tensorboard_callback, ComputeMetrics(), csv_logger],
            workers=config.cl_dataloader_workers,
        )

    def print_model_and_class_values(self):
        sprint(
            self.train_data_folder,
            self.validation_data_folder,
            self.shape,
            self.validation_csv_file,
            self.log_dir,
            self.model,
            self.model.summary(),
        )


if __name__ == "__main__":
    model = ConvLSTM(
        training_data_folder="training_data_8_4_8",
        validation_data_folder="validation_data_8_4_8",
        shape=(2, 8, 32, 32, 1),
        validation_csv_file="validation.csv",
        log_dir="log_dir",
    )
    model.print_model_and_class_values()
    model.train()
