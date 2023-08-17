import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file

from complexity.complexityUNET import Complexity

import config

from validation_metrics.custom_losses import non_zero_mape, non_zero_mse

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from preprocessing.datagen_for_Unet import CustomDataGenerator

from smartprint import smartprint as sprint
from tqdm import tqdm


import numpy as np
import tensorflow
import os
import glob

# Create CSVLogger callback with specified filename
from tensorflow.keras.callbacks import Callback
from baselines.NaiveBaseline import NaiveBaselineUnet as NaiveBaseline
from preprocessing.ProcessRaw import ProcessRaw



class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 5 == 0 or True: # Always run for the case of Unet
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


        logs["naive-model-non-zero_Unet"] = (
            NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)
        ).naive_baseline_mse_non_zero
        logs["naive-model-mse_Unet"] = (NaiveBaseline(1, 1).from_dataloader(self.model.train_gen, 50)).naive_baseline_mse

        # save the model to disk
        if config.cl_model_save_epoch_end:
            self.model.save(
                os.path.join(
                    config.INTERMEDIATE_FOLDER,
                    os.path.basename(os.path.normpath(self.model.prefix)) + "_epoch_Unet_" + str(epoch) + ".h5",
                )
            )

class Unet:
    def __init__(self, cityname, io_length, pred_horiz, scale, log_dir, shape, validation_csv_file, input_shape=(84, 84, 4),
                 saved_model_filename=None):
        """
        Input and output shapes are the same (tuple of length 5)
        """
        self.cityname, self.io_length, self.pred_horiz, self.scale = cityname, io_length, pred_horiz, scale
        self.prefix = ProcessRaw.file_prefix(cityname, io_length, pred_horiz, scale)

        self.saved_model_filename = saved_model_filename
        self.train_data_folder = os.path.join(config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, self.prefix)
        self.validation_data_folder = os.path.join(config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, self.prefix)
        self.shape = shape
        self.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, validation_csv_file)
        sprint(validation_csv_file, self.validation_csv_file)
        self.log_dir = os.path.join(config.INTERMEDIATE_FOLDER, log_dir)
        self.model = self.create_model(input_shape)

    def create_model(self, input_shape=(54, 54, 4)):
        # Pad the input to 56x56x4
        inputs = layers.Input((54, 54, 4))
        padded_input = layers.ZeroPadding2D(((1, 1), (1, 1)))(inputs)
        print(f"Padded input shape: {padded_input.shape}")

        # Encoder
        c1 = (layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(padded_input)
        print(f"c1 shape: {c1.shape}")
        p1 = (layers.MaxPooling2D((2, 2)))(c1)
        print(f"p1 shape: {p1.shape}")
        c2 = (layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(p1)
        print(f"c2 shape: {c2.shape}")
        p2 = (layers.MaxPooling2D((2, 2)))(c2)
        print(f"p2 shape: {p2.shape}")

        # Middle
        c3 = (layers.Conv2D(256, (3, 3), activation='relu', padding='same'))(p2)
        print(f"c3 shape: {c3.shape}")

        # Decoder
        u1 = (layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same'))(c3)
        print(f"u1 shape: {u1.shape}")
        c4 = (layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(u1)
        print(f"c4 shape: {c4.shape}")
        u2 = (layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same'))(c4)
        print(f"u2 shape: {u2.shape}")
        c5 = (layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(u2)
        print(f"c5 shape: {c5.shape}")

        # Output
        outputs = (layers.Conv2D(4, (1, 1), activation='sigmoid'))(c5)
        print(f"outputs shape: {outputs.shape}")

        # Crop the output to 54x54x4
        outputs_cropped = layers.Cropping2D(((1, 1), (1, 1)))(outputs)
        print(f"Cropped outputs shape: {outputs_cropped.shape}")

        self.model = tensorflow.keras.Model(inputs=inputs, outputs=outputs_cropped)
        return self.model

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
        self.model.validation_gen = self.validation_gen
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
            for x, y in self.model.train_gen:
                print ("Shape of x, y from train_gen before calling fit")
                sprint (x.shape, y.shape)
                break

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
                run_pm=False,
                run_nm=False,
                run_gb=False,
            )
            print(
                "TRAIN_END: ",
                self.prefix,
                self.cityname,
                self.io_length,
                self.pred_horiz,
                self.scale,
                cx.CSR_MP_sum_y_exceeding_r_x_max
            )
        elif config.cl_post_model_loading_from_saved_val_error_plots_spatial_or_temporal:
            cx = Complexity(
                self.model.cityname,
                i_o_length=self.model.io_length,
                prediction_horizon=self.model.pred_horiz,
                grid_size=self.model.scale,
                thresh=config.cl_thresh,
                perfect_model=False,
                model_func=self.model.predict,
                model_train_gen=self.model.validation_gen,
                # only one line change compared to the standard call above
                run_pm=False,
                run_nm=False,
                run_gb=False,
            )
            print(
                "TRAIN_END: ",
                self.prefix,
                self.cityname,
                self.io_length,
                self.pred_horiz,
                self.scale,
                cx.CSR_MP_sum_y_exceeding_r_x_max,
            )
        # no need to train anymore
        return self.model

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
    io_length = 4
    pred_horiz = 1
    scale = 55

    obj = ProcessRaw(cityname=cityname, i_o_length=io_length, prediction_horizon=pred_horiz, grid_size=scale)

    model = Unet(
        cityname,
        io_length,
        pred_horiz,
        scale,
        shape=(2, io_length, scale, scale, 1),
        validation_csv_file="unet_validation.csv",
        log_dir="unet_log_dir",
    )
    print(model.model.summary())
    model.print_model_and_class_values(print_model_summary=False)
    model.train()

    # Now, we can delete the temp files after training for one scenario
    # obj._clean_intermediate_files()
