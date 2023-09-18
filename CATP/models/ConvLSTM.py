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
import shutil

class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        logs["naive-model-non-zero"] = (
            NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)
        ).naive_baseline_mse_non_zero
        logs["naive-model-mse"] = (NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)).naive_baseline_mse

        # save the model to disk
        if config.cl_model_save_epoch_end:
            self.model.save(
                os.path.join(
                    config.INTERMEDIATE_FOLDER,
                    os.path.basename(os.path.normpath(self.model.prefix)) + "_epoch_" + str(epoch) + ".h5",
                )
            )

        # Clean directory
        predictions_dir = self.model.predictions_folder
        sprint (predictions_dir)
        if os.path.exists(predictions_dir):
            shutil.rmtree(predictions_dir)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)

        for index in range(len(self.model.train_gen)):
            x_batch, _, indexes = self.model.train_gen.get_item_with_indexes(index)
            predictions = self.model.predict(x_batch)

            for idx, (input_data, pred) in enumerate(zip(x_batch, predictions)):
                # Save the input data and the prediction into the corect prediction folder
                input_file = os.path.join(predictions_dir, 
                                          "{}{}_x.npy".format(self.model.train_gen.prefix, indexes[idx]))
                temp_input_file_name = str(int(np.random.rand() * 100000000000)) + "_x.npy"
                np.save(temp_input_file_name, input_data)
                os.rename(temp_input_file_name, input_file)

                # Save the prediction
                predictions_file = os.path.join(predictions_dir,
                                                "{}{}_y.npy".format(self.model.train_gen.prefix, indexes[idx]))
                temp_pred_file_name = str(int(np.random.rand() * 100000000000)) + "_y.npy"
                np.save(temp_pred_file_name, pred)
                os.rename(temp_pred_file_name, predictions_file)
                
        # if epoch % 0 == 0:

        if config.cl_during_training_CSR_enabled_epoch_end:
            cx = Complexity(
                self.model.cityname,
                i_o_length=self.model.io_length,
                prediction_horizon=self.model.pred_horiz,
                grid_size=self.model.scale,
                perfect_model=False,
                model_func=self.model.predict,
                model_train_gen=self.model.train_gen,
                model_predict_gen=self.model.predict_gen,
                run_pm=True,
                run_nm=True,
                run_gb=True,
            )

            logs["CSR_MP_sum_y_exceeding_r_x_max"] = cx.CSR_MP_sum_y_exceeding_r_x_max
            logs["CSR_PM_sum_y_exceeding_r_x_max"] = cx.CSR_PM_sum_y_exceeding_r_x_max
            logs["CSR_NM_sum_y_exceeding_r_x_max"] = cx.CSR_NM_sum_y_exceeding_r_x_max
            logs["CSR_GB_sum_y_exceeding_r_x_max"] = cx.CSR_GB_sum_y_exceeding_r_x_max


        else:
            logs["CSR_MP_sum_y_exceeding_r_x_max"] = -1
            logs["CSR_PM_sum_y_exceeding_r_x_max"] = -1
            logs["CSR_NM_sum_y_exceeding_r_x_max"] = -1
            logs["CSR_GB_sum_y_exceeding_r_x_max"] = -1



class ConvLSTM:
    def __init__(self, cityname, io_length, pred_horiz, scale, log_dir, shape, validation_csv_file, saved_model_filename=None):
        """
        Input and output shapes are the same (tuple of length 5)
        """
        self.cityname, self.io_length, self.pred_horiz, self.scale = cityname, io_length, pred_horiz, scale
        self.prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)

        self.saved_model_filename = saved_model_filename
        self.train_data_folder = os.path.join(config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, self.prefix)
        self.validation_data_folder = os.path.join(config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, self.prefix)
        self.predictions_folder = os.path.join(config.HOME_FOLDER, "predictions_folder", self.prefix)

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
            filters=128,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
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

    def create_model_small_epochs(self):
        _, a, b, c, d = self.shape
        x = np.random.rand(2, a, b, c, d)
        inp = layers.Input(shape=(None, *x.shape[2:]))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = layers.ConvLSTM2D(
            filters=4,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=4,
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

    def create_model_dummy_single_layer(self):
        _, a, b, c, d = self.shape
        x = np.random.rand(2, a, b, c, d)
        inp = layers.Input(shape=(None, *x.shape[2:]))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = layers.ConvLSTM2D(
            filters=4,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
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
        self.prediction_gen = CustomDataGenerator(
            self.cityname,
            self.io_length,
            self.pred_horiz,
            self.scale,
            data_dir=self.predictions_folder,
            num_samples=int(num_train * r),
            batch_size=batch_size,
            shuffle=True,
        )

        csv_logger = CSVLogger(self.validation_csv_file)
        sprint(csv_logger, self.validation_data_folder)
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=self.log_dir)

        self.model.train_gen = self.train_gen
        self.model.validation_gen = self.validation_gen
        self.model.predict_gen = self.prediction_gen
        self.model.predictions_folder = self.predictions_folder
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

        if self.saved_model_filename is not None:
            # Loading saved model is written inside train instead of create model since
            # our callback classes/ params need to be in the environment before loading
            # otherwise it fails to load
            self.model.load_weights(self.saved_model_filename)

            # no need to train in that case
        else:
            self.model.fit(
                self.train_gen,
                validation_data=self.validation_gen,
                epochs=epochs,
                callbacks=callbacks,
                workers=config.cl_dataloader_workers,
            )
        return self.model


    def print_model_and_class_values(self):
        sprint(self.train_data_folder)
        sprint(self.validation_data_folder)
        sprint(self.shape)
        sprint(self.validation_csv_file)
        sprint(self.log_dir)
        sprint(self.model)
        sprint(self.prefix)

    def __repr__(self):
        attributes = [
            f"train_data_folder={self.train_data_folder}",
            f"validation_data_folder={self.validation_data_folder}",
            f"shape={self.shape}",
            f"validation_csv_file={self.validation_csv_file}",
            f"log_dir={self.log_dir}",
            f"model={self.model}",
            # This might not provide a meaningful string representation depending on what self.model is
            f"prefix={self.prefix}"
        ]

        return "<ConvLSTM " + "\n".join(attributes) + ">"

    @staticmethod
    def test_ConvLSTM():
        obj = ProcessRaw(cityname=config.city_list_def[0], i_o_length=config.i_o_lengths_def[0],
                         prediction_horizon=config.pred_horiz_def[0], grid_size=config.scales_def[0])

        model = ConvLSTM(
            config.city_list_def[0],
            config.i_o_lengths_def[0],
            config.pred_horiz_def[0],
            config.scales_def[0],
            shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
            validation_csv_file=obj.key_dimensions() + "validation.csv",
            log_dir=obj.key_dimensions() + "log_dir",
        )
        print (model.model.summary())
        print (model)
        model.train()


if __name__ == "__main__":
    # ConvLSTM.test_ConvLSTM()
    for city in config.city_list_def:
        for pred_horiz in config.pred_horiz:
            obj = ProcessRaw(cityname=city, i_o_length=config.i_o_lengths_def[0],
                             prediction_horizon=pred_horiz, grid_size=config.scales_def[0])

            model = ConvLSTM(
                city,
                config.i_o_lengths_def[0],
                pred_horiz,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
            )
            print(model.model.summary())
            print(model)
            model.train()

