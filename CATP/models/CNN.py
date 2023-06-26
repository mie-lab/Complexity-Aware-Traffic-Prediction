import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from validation_metrics.custom_losses import non_zero_mape, non_zero_mse

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from preprocessing.datagen import CustomDataGenerator
from complexity.complexity import Complexity
from smartprint import smartprint as sprint
from tqdm import tqdm


import numpy as np
import tensorflow
import os
import glob

# Create CSVLogger callback with specified filename
from tensorflow.keras.callbacks import Callback
from baselines.NaiveBaseline import NaiveBaseline
from preprocessing.ProcessRaw import ProcessRaw


class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        if config.cl_during_training_CSR_enabled_epoch_end:
            cx = Complexity(
                self.model.cityname,
                i_o_length=self.model.io_length,
                prediction_horizon=self.model.pred_horiz,
                grid_size=self.model.scale,
                thresh=config.cl_thresh,
                perfect_model=False,
                model_func=self.model.predict,
                model_train_gen=self.model.train_gen,
            )
            logs["CSR_train_data_DL_epoch_end"] = cx.CSR_MP_no_thresh_mean
        else:
            logs["CSR_train_data_DL_epoch_end"] = 1

        logs["naive-model-non-zero"] = (
            NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)
        ).naive_baseline_mse_non_zero
        logs["naive-model-mse"] = (NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)).naive_baseline_mse

        # save the model to disk
        if config.cl_model_save:
            self.model.save(
                os.path.join(
                    config.INTERMEDIATE_FOLDER,
                    os.path.basename(os.path.normpath(self.model.prefix)) + "_epoch_" + str(epoch) + ".h5",
                )
            )

    # def on_train_end(self, logs):
    #     if config.cl_during_training_CSR_enabled_train_end:
    #         cx = Complexity(self.model.cityname, i_o_length=self.model.io_length,
    #                         prediction_horizon=self.model.pred_horiz, \
    #                         grid_size=self.model.scale, thresh=config.cl_thresh, perfect_model=False, \
    #                         model_func=self.model.predict, model_train_gen=self.model.train_gen)
    #         logs["CSR_train_data_DL_train_end"] = cx.CSR_MP_no_thresh_mean
    #     else:
    #         logs["CSR_train_data_DL_train_end"] = 1


class CNN:
    def __init__(self, cityname, io_length, pred_horiz, scale, log_dir, shape, validation_csv_file):
        """
        Input and output shapes are the same (tuple of length 5)
        """
        self.cityname, self.io_length, self.pred_horiz, self.scale = cityname, io_length, pred_horiz, scale
        self.prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)

        self.train_data_folder = os.path.join(config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, self.prefix)
        self.validation_data_folder = os.path.join(config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, self.prefix)
        self.shape = shape
        self.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, validation_csv_file)
        sprint(validation_csv_file, self.validation_csv_file)
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
        # model.compile(
        #     loss=tensorflow.keras.losses.binary_crossentropy,
        #     optimizer=tensorflow.keras.optimizers.Adam(),
        # )

        return model

    def train(self):
        # Train the model
        batch_size = config.cl_batch_size
        epochs = config.cl_epochs
        if config.cl_loss_func == "mse":
            loss_fn = config.cl_loss_func
        elif config.cl_loss_func == "non-zero-mse":
            loss_fn = non_zero_mse

        optimizer = optimizers.Adam(1e-3)

        # manual reset the model since sometimes it does not reset a new model (not sure why)
        tensorflow.keras.backend.clear_session()
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=non_zero_mse)

        num_train = len(
            glob.glob(os.path.join(config.HOME_FOLDER, self.train_data_folder) + "/" + self.prefix + "*_x.npy")
        )
        num_validation = len(
            glob.glob(os.path.join(config.HOME_FOLDER, self.validation_data_folder) + "/" + self.prefix + "*_x.npy")
        )

        r = config.cl_percentage_of_train_data  # np.random.rand()

        self.train_gen = CustomDataGenerator(
            self.cityname,
            self.io_length,
            self.pred_horiz,
            self.scale,
            data_dir=self.train_data_folder,
            num_samples=int(num_train * r),
            batch_size=batch_size,
            shuffle=True,
        )
        self.validation_gen = CustomDataGenerator(
            self.cityname,
            self.io_length,
            self.pred_horiz,
            self.scale,
            data_dir=self.validation_data_folder,
            num_samples=int(num_validation * r),
            batch_size=batch_size,
            shuffle=True,
        )

        csv_logger = CSVLogger(self.validation_csv_file)
        sprint(csv_logger, self.validation_data_folder)
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=self.log_dir)

        self.model.train_gen = self.train_gen
        self.model.prefix = self.prefix
        self.model.cityname, self.model.io_length, self.model.pred_horiz, self.model.scale = (
            self.cityname,
            self.io_length,
            self.pred_horiz,
            self.scale,
        )

        callbacks = []
        if config.cl_early_stopping_patience != -1:
            earlystop = EarlyStopping(
                monitor="val_loss",
                patience=config.cl_early_stopping_patience,
                verbose=0,
                mode="auto",
                restore_best_weights=True,
            )
            callbacks.append(earlystop)

        if config.cl_tensorboard:
            callbacks.extend([tensorboard_callback, ComputeMetrics(), csv_logger])
        else:
            callbacks.extend([ComputeMetrics(), csv_logger])

        self.model.fit(
            self.train_gen,
            validation_data=self.validation_gen,
            epochs=epochs,
            callbacks=callbacks,
            workers=config.cl_dataloader_workers,
        )

        if config.cl_during_training_CSR_enabled_train_end:
            cx = Complexity(
                self.model.cityname,
                i_o_length=self.model.io_length,
                prediction_horizon=self.model.pred_horiz,
                grid_size=self.model.scale,
                thresh=config.cl_thresh,
                perfect_model=False,
                model_func=self.model.predict,
                model_train_gen=self.model.train_gen,
            )
            print(
                "TRAIN_END: ",
                self.prefix,
                self.cityname,
                self.io_length,
                self.pred_horiz,
                self.scale,
                cx.CSR_MP_no_thresh_mean,
                cx.CSR_PM_no_thresh_frac_mean,
                cx.CSR_PM_no_thresh_mean,
            )

    def print_model_and_class_values(self, print_model_summary=True):
        sprint(self.train_data_folder)
        sprint(self.validation_data_folder)
        sprint(self.shape)
        sprint(self.validation_csv_file)
        sprint(self.log_dir)
        sprint(self.model)
        sprint(self.prefix)
        if print_model_summary:
            sprint(self.model.summary())


if __name__ == "__main__":
    cityname = "london"
    io_length = 3
    pred_horiz = 4
    scale = 5

    obj = ProcessRaw(cityname=cityname, i_o_length=io_length, prediction_horizon=pred_horiz, grid_size=scale)

    model = ConvLSTM(
        cityname,
        io_length,
        pred_horiz,
        scale,
        shape=(2, io_length, scale, scale, 1),
        validation_csv_file="validation.csv",
        log_dir="log_dir",
    )
    print(model.model.summary())
    model.print_model_and_class_values(print_model_summary=False)
    model.train()
    # model.predict_train_data_and_save_all()

    # Now, we can delete the temp files after training for one scenario
    obj.clean_intermediate_files()
