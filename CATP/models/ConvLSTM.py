import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file
import config

from validation_metrics.custom_losses import non_zero_mape, non_zero_mse

from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from preprocessing.datagen import CustomDataGenerator
from complexity.complexity import Complexity
from smartprint import smartprint as sprint
from tqdm import tqdm
import inspect

from slugify import slugify

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
                    os.path.basename(os.path.normpath(self.model.prefix)) + "_epoch_" + slugify(self.model.predictions_folder)  +  "_" + str(epoch) + ".h5",
                )
            )

        # Clean directory
        predictions_dir = self.model.predictions_folder
        sprint (predictions_dir)
        if os.path.exists(predictions_dir):
            shutil.rmtree(predictions_dir)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)


        if config.RUNNING_IC_TEMP:
            model_gen = self.model.validation_gen
        else:
            model_gen = self.model.train_gen

        for index in range(len(model_gen)): # train_gen to be replaced with validation_gen when  running spatial or temporal
            # Total changes in 4 places
            x_batch, _, indexes = model_gen.get_item_with_indexes(index) # train_gen to be replaced with validation_gen when  running spatial or temporal
            predictions = self.model.predict(x_batch)


            for idx, (input_data, pred) in enumerate(zip(x_batch, predictions)):
                # Save the input data and the prediction into the correct prediction folder
                input_file = os.path.join(predictions_dir,
                                          "{}{}_x.npy".format(model_gen.prefix, indexes[idx])) #  # train_gen to be replaced with validation_gen when  running spatial or temporal
                temp_input_file_name = str(int(np.random.rand() * 100000000000)) + "_x.npy"                 # since in that case we need the val MSE per sample
                                                                                                            # hence the predicitons should be using the valdation data
                np.save(temp_input_file_name, input_data)
                os.rename(temp_input_file_name, input_file)

                # Save the prediction
                predictions_file = os.path.join(predictions_dir,
                                                "{}{}_y.npy".format(model_gen.prefix, indexes[idx]))  # train_gen to be replaced with pred_gen when  running spatial or temporal
                temp_pred_file_name = str(int(np.random.rand() * 100000000000)) + "_y.npy"
                np.save(temp_pred_file_name, pred)
                os.rename(temp_pred_file_name, predictions_file)

        # if epoch % 0 == 0:
        if epoch % 5 == 0: # To be used when running spatial or temporal


        # if epoch == 1: # To ensure that we compute the ValMSE only for the last epoch when running the IC_temp
                        # this value should be equal to the last epoch;

            RUN_PM=True
        else:
            RUN_PM = False

        compute_CX = False
        if config.RUNNING_IC_TEMP:
            if RUN_PM and config.cl_during_training_CSR_enabled_epoch_end: # To be used when running spatial or temporal
                                                                        # experiment, since we want to compute the Val-MSE
                compute_CX = True
        elif not config.RUNNING_IC_TEMP:
            if config.cl_during_training_CSR_enabled_epoch_end:    # for all other cases
                compute_CX = True

        if compute_CX:
            cx = Complexity(
                self.model.cityname,
                i_o_length=self.model.io_length,
                prediction_horizon=self.model.pred_horiz,
                grid_size=self.model.scale,
                perfect_model=False,
                model_func=self.model.predict,
                model_train_gen=self.model.train_gen,
                model_predict_gen=self.model.predict_gen,
                run_pm=RUN_PM,
                run_nm=True,
                run_gb=True,
                predictions_dir=self.model.predictions_folder,
            )

            logs["CSR_MP_sum"] = cx.CSR_MP_sum_y_exceeding_r_x_max
            logs["CSR_PM_sum"] = cx.CSR_PM_sum_y_exceeding_r_x_max
            logs["CSR_NM_sum"] = cx.CSR_NM_sum_y_exceeding_r_x_max
            logs["CSR_GB_sum"] = cx.CSR_GB_sum_y_exceeding_r_x_max
            logs["CSR_MP_std"] = cx.CSR_MP_std
            logs["CSR_PM_std"] = cx.CSR_PM_std
            logs["CSR_MP_count"], logs["CSR_PM_count"], logs["CSR_NM_count"], logs["CSR_GB_count"]\
                = cx.CSR_MP_count, cx.CSR_PM_count, cx.CSR_NM_count, cx.CSR_GB_count
        else:
            logs["CSR_MP_sum"] = -1
            logs["CSR_PM_sum"] = -1
            logs["CSR_NM_sum"] = -1
            logs["CSR_GB_sum"] = -1
            logs["CSR_MP_std"] = -1
            logs["CSR_PM_std"] = -1
            logs["CSR_MP_count"], logs["CSR_PM_count"], logs["CSR_NM_count"], logs["CSR_GB_count"] \
                = -1, -1, -1, -1

class CustomModel(tensorflow.keras.models.Model):
    def test_step(self, data):
        x, y, _ = data
        y_pred = self(x, training=False)
        self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tensorflow.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

class ConvLSTM:
    def __init__(self, cityname, io_length, pred_horiz, scale, log_dir, shape, validation_csv_file, saved_model_filename=None, custom_eval=False):
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
        self.model = self.create_model(custom_eval)

    def create_model(self, custom_eval=False):
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
        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)
        # model.compile(
        #     loss=tensorflow.keras.losses.binary_crossentropy,
        #     optimizer=tensorflow.keras.optimizers.Adam(),
        # )

        return model

    def create_model_flexible(self, depth, num_filters, custom_eval=False, BN=False, KR=False):
        _, a, b, c, d = self.shape
        x_input_shape = np.random.rand(2, a, b, c, d).shape
        inp = layers.Input(shape=(None, *x_input_shape[2:]))

        x = inp
        for _ in range(depth):
            if KR:
                reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)  # L1L2 regularization
                x = layers.ConvLSTM2D(
                    filters=num_filters,
                    kernel_size=(3, 3),
                    padding="same",
                    return_sequences=True,
                    activation="relu",
                    kernel_regularizer=reg,
                    bias_regularizer=reg
                )(x)
            else:
                x = layers.ConvLSTM2D(
                    filters=num_filters,
                    kernel_size=(3, 3),
                    padding="same",
                    return_sequences=True,
                    activation="relu",
                )(x)
                if BN:
                    x = layers.BatchNormalization()(x)


        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)

        return model

    def create_model_f_def_no_BN(self, custom_eval=False):
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
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        # Next, we will build the complete model and compile it.
        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)
        # model.compile(
        #     loss=tensorflow.keras.losses.binary_crossentropy,
        #     optimizer=tensorflow.keras.optimizers.Adam(),
        # )

        return model

    def create_model_f_big_no_BN(self, custom_eval=False):
        _, a, b, c, d = self.shape
        inp = layers.Input(shape=(None, b, c, d))

        x = layers.ConvLSTM2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.ConvLSTM2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
        # x = layers.Conv3D(filters=128, kernel_size=(1, 1, 1), activation="relu", padding="same")(x)


        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)

        return model

    def create_model_f_shallow_1_no_BN(self, custom_eval=False):
        _, a, b, c, d = self.shape
        inp = layers.Input(shape=(None, b, c, d))

        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)

        return model

    def create_model_f_shallow_2_no_BN(self, custom_eval=False):
        _, a, b, c, d = self.shape
        inp = layers.Input(shape=(None, b, c, d))

        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)

        return model

    def create_model_f_shallow_3_no_BN(self, custom_eval=False):
        _, a, b, c, d = self.shape
        inp = layers.Input(shape=(None, b, c, d))

        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)

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

    def create_model_small_epochs_no_BN(self):
        _, a, b, c, d = self.shape
        x = np.random.rand(2, a, b, c, d)
        inp = layers.Input(shape=(None, *x.shape[2:]))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = layers.ConvLSTM2D(
            filters=16,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.ConvLSTM2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=8,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.ConvLSTM2D(
            filters=8,
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


    def train(self, epochs_param=-1, optim="Adam"):
        # Train the model
        batch_size = config.cl_batch_size
        if epochs_param == -1:
            epochs = config.cl_epochs
        else:
            epochs = epochs_param
        if config.cl_loss_func == "mse":
            loss_fn = config.cl_loss_func
        elif config.cl_loss_func == "non-zero-mse":
            loss_fn = non_zero_mse

        if optim=="Adam":
            optimizer = optimizers.Adam(0.001) #, 0.6, 0.9)
        elif optim=="SGD":
            optimizer = optimizers.SGD(0.1)
        else:
            raise Exception("Wrong optimiser provided")

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
            batch_size=config.cl_prediction_batch_size,
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

    @staticmethod
    def experiment_simple():
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

    @staticmethod
    def experiment_mix_examples_control():
        for city in config.city_list_def:
            for pred_horiz in [1]:
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
                    custom_eval=False
                )
                assert config.cx_sampling_enabled == False
                print(model.model.summary())

                # We dont train on the easy case
                model.train(30)

    @staticmethod
    def experiment_mix_examples_exp_sampling():
        for city in config.city_list_def:
            for pred_horiz in [1]:
                obj = ProcessRaw(cityname=city, i_o_length=config.i_o_lengths_def[0],
                                 prediction_horizon=pred_horiz, grid_size=config.scales_def[0])

                model = ConvLSTM(
                    city,
                    config.i_o_lengths_def[0],
                    pred_horiz,
                    config.scales_def[0],
                    shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                    validation_csv_file=obj.key_dimensions() + "validation_sampling_case.csv",
                    log_dir=obj.key_dimensions() + "log_dir",
                    custom_eval=True
                )
                print(model.model.summary())

                # We dont train on the easy case
                model.train(30)


    @staticmethod
    def experiment_mix_pred_horiz_2_1():
        for city in config.city_list_def:
            for pred_horiz in [2]:
                obj = ProcessRaw(cityname=city, i_o_length=config.i_o_lengths_def[0],
                                 prediction_horizon=pred_horiz, grid_size=config.scales_def[0])

                model = ConvLSTM(
                    city,
                    config.i_o_lengths_def[0],
                    pred_horiz,
                    config.scales_def[0],
                    shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                    validation_csv_file=obj.key_dimensions() + "validation_pred_horiz_2.csv",
                    log_dir=obj.key_dimensions() + "log_dir_pred_horiz_2",
                    custom_eval=True 
                )
                model = ConvLSTM.helper_switch_datasets(model, 1, "validation_pred_horiz_1.csv", -1, city, train=False)
                print(model.model.summary())
                # train only on the hard case
                model.train(30)

    @staticmethod
    def experiment_mix_examples():
        for city in config.city_list_def:
            for pred_horiz in [1]:
                obj = ProcessRaw(cityname=city, i_o_length=config.i_o_lengths_def[0],
                                 prediction_horizon=pred_horiz, grid_size=config.scales_def[0])


                model = ConvLSTM(
                    city,
                    config.i_o_lengths_def[0],
                    pred_horiz,
                    config.scales_def[0],
                    shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                    validation_csv_file=obj.key_dimensions() + "val_file_exp.csv",
                    log_dir=obj.key_dimensions() + "log_dir_exp",
                )
                print(model.model.summary())

                NEW_PRED_HORIZ = 6

                # Define the data to loop over
                configs = [
                    (NEW_PRED_HORIZ, "validation_csv_hard_1.csv"),
                    (pred_horiz, "validation_csv_easy_1.csv"),
                    (NEW_PRED_HORIZ, "validation_csv_hard_2.csv"),
                    (pred_horiz, "validation_csv_easy_2.csv"),
                    (NEW_PRED_HORIZ, "validation_csv_hard_3.csv"),
                    (pred_horiz, "validation_csv_easy_3.csv"),
                    (NEW_PRED_HORIZ, "validation_csv_hard_4.csv"),
                    (pred_horiz, "validation_csv_easy_4.csv"),
                    (NEW_PRED_HORIZ, "validation_csv_hard_5.csv"),
                    (pred_horiz, "validation_csv_easy_5.csv")
                ]

                # Loop through the configurations and call the function
                for pred_horiz, val_file in configs:
                    model = ConvLSTM.helper_switch_datasets(
                                        model=model,
                                           NEW_PRED_HORIZ=pred_horiz,
                                           VAL_file_NAME_SWITCHED=val_file,
                                           Epochs=2,
                                           city=city,
                                        )

    @staticmethod
    def helper_switch_datasets(model, NEW_PRED_HORIZ,VAL_file_NAME_SWITCHED, Epochs, city, train=True):
        obj = ProcessRaw(cityname=city, i_o_length=config.i_o_lengths_def[0],
                             prediction_horizon=NEW_PRED_HORIZ, grid_size=config.scales_def[0])

        new_name = ProcessRaw.file_prefix(city, config.i_o_lengths_def[0], NEW_PRED_HORIZ, config.scales_def[0])
        self_new_name_train_data_folder = os.path.join(config.DATA_FOLDER, config.TRAINING_DATA_FOLDER, new_name)
        self_new_name_val_data_folder = os.path.join(config.DATA_FOLDER, config.VALIDATION_DATA_FOLDER, new_name)
        self_new_name_predictions_folder = os.path.join(config.HOME_FOLDER, "predictions_folder_exp", new_name)
        num_train = len(
            glob.glob(self_new_name_train_data_folder + "/" + new_name + "*_x.npy")
        )
        num_validation = len(
            glob.glob(self_new_name_val_data_folder + "/" + new_name + "*_x.npy")
        )

        model.predictions_folder = self_new_name_predictions_folder
        model.train_data_folder = self_new_name_train_data_folder
        model.validation_data_folder = self_new_name_val_data_folder

        r = config.cl_percentage_of_train_data
        batch_size = config.cl_batch_size
        model.train_gen = CustomDataGenerator(
            model.cityname,
            model.io_length,
            NEW_PRED_HORIZ,
            model.scale,
            data_dir=self_new_name_train_data_folder,
            num_samples=int(num_train * r),
            batch_size=batch_size,
            shuffle=True,
        )

        model.prediction_gen = CustomDataGenerator(
            model.cityname,
            model.io_length,
            NEW_PRED_HORIZ,
            model.scale,
            data_dir=self_new_name_predictions_folder,
            num_samples=int(num_train * r),
            batch_size=batch_size,
            shuffle=True,
        )

        model.validation_gen = CustomDataGenerator(
            model.cityname,
            model.io_length,
            NEW_PRED_HORIZ,
            model.scale,
            data_dir=self_new_name_val_data_folder,
            num_samples=int(num_validation * r),
            batch_size=batch_size,
            shuffle=True,
        )
        model.log_dir = os.path.join(config.INTERMEDIATE_FOLDER, obj.key_dimensions() +
                                     "log_dir_exp_switched")
        model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, obj.key_dimensions() +
                                                 VAL_file_NAME_SWITCHED)

        # Make sure everything is passed correctly to the callback functions before calling model.train()
        model.model.validation_gen = model.validation_gen
        model.model.train_gen = model.train_gen
        model.model.predict_gen = model.prediction_gen
        model.model.predictions_folder = model.predictions_folder
        model.model.prefix = new_name
        model.prefix = model.model.prefix
        model.pred_horiz = NEW_PRED_HORIZ
        model.model.cityname, model.model.io_length, model.model.pred_horiz, model.model.scale = (
            model.cityname,
            model.io_length,
            model.pred_horiz,
            model.scale,
        )
        if train:
            model.train(Epochs)
        return model

    @staticmethod
    def print_all_model_summary():
        obj = ProcessRaw(cityname="london", i_o_length=4,
                         prediction_horizon=4, grid_size=55)

        model = ConvLSTM(
            "london",
            4,
            4,
            55,
            shape=(2, 4, 55, 55, 1),
            validation_csv_file=obj.key_dimensions() + "validation_sampling_case.csv",
            log_dir=obj.key_dimensions() + "log_dir",
            custom_eval=True
        )
        list_of_models = ConvLSTM.get_methods_of_class(ConvLSTM)
        list_of_models = [x for x in list_of_models if "create_model" in x]
        for model_type in list_of_models:
            new_model = getattr(model, model_type)()
            print (model_type)
            print (new_model.summary())


    @staticmethod
    def one_task_different_models():
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
            custom_eval=False
        )

        for filters in [32, 64, 128]:
            for depth in [1, 2, 4]:
                model.model = model.create_model_flexible(depth=depth, num_filters=filters,
                                                          custom_eval=False, BN=True)

                model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-DEP-" + str(depth)
                                                         + "-FIL-" + str(filters) +
                                                         "-adam-01-" + slugify("one_task_different_models") +
                                                         obj.key_dimensions() + ".csv")
                # print (updated_model.summary())

                from contextlib import redirect_stdout

                with open('modelsummary.txt', 'w') as f:
                    with redirect_stdout(f):
                        model.model.summary()

                print ("======================================")
                sprint (obj.key_dimensions(), "DEP-" + str(depth) + "-FIL-" + str(filters))
                os.system("grep \'Trainable params:\' modelsummary.txt")

                # print (model.model.summary())
                # model.train(epochs_param=30, optim="Adam")

    @staticmethod
    def one_task_one_model_with_and_without_lr():
        for PRED_HORIZ in range(1, 2):
            obj = ProcessRaw(cityname=config.city_list_def[0], i_o_length=config.i_o_lengths_def[0],
                             prediction_horizon=PRED_HORIZ, grid_size=config.scales_def[0])

            model = ConvLSTM(
                config.city_list_def[0],
                config.i_o_lengths_def[0],
                PRED_HORIZ,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            # BN_True
            model.model = model.create_model_flexible(depth=3, num_filters=64,
                                                              custom_eval=False, BN=True, KR=True)


            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-001-" + slugify("-one-task-one-model-L1l2-reg-") + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.model.summary()

            print("======================================")
            os.system("grep \'Trainable params:\' modelsummary.txt")

            # print (model.model.summary())
            model.train(epochs_param=30, optim="Adam")


            model = ConvLSTM(
                config.city_list_def[0],
                config.i_o_lengths_def[0],
                PRED_HORIZ,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            # BN_True
            model.model = model.create_model_flexible(depth=3, num_filters=64,
                                                              custom_eval=False, BN=True, KR=False)


            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-001-" + slugify("-one-task-one-model-no-L1l2-reg-") + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.model.summary()

            print("======================================")
            os.system("grep \'Trainable params:\' modelsummary.txt")

            # print (model.model.summary())
            model.train(epochs_param=30, optim="Adam")


    @staticmethod
    def one_task_one_model_different_BN_no_BN():
        for PRED_HORIZ in range(2, 5):
            obj = ProcessRaw(cityname=config.city_list_def[0], i_o_length=config.i_o_lengths_def[0],
                             prediction_horizon=PRED_HORIZ, grid_size=config.scales_def[0])

            model = ConvLSTM(
                config.city_list_def[0],
                config.i_o_lengths_def[0],
                PRED_HORIZ,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            # BN_True
            model.model = model.create_model_flexible(depth=3, num_filters=64,
                                                              custom_eval=False, BN=True)


            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p001-" + slugify("-one-task-one-model-BN-") + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.model.summary()

            print("======================================")
            os.system("grep \'Trainable params:\' modelsummary.txt")

            # print (model.model.summary())
            model.train(epochs_param=30, optim="Adam")

            model = ConvLSTM(
                config.city_list_def[0],
                config.i_o_lengths_def[0],
                PRED_HORIZ,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            # BN_True
            model.model = model.create_model_flexible(depth=3, num_filters=64,
                                                      custom_eval=False, BN=False)

            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                     "-adam-0p001-" + slugify(
                "-one-task-one-model-no-BN-") + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.model.summary()

            print("======================================")
            os.system("grep \'Trainable params:\' modelsummary.txt")

            # print (model.model.summary())
            model.train(epochs_param=30, optim="Adam")



    @staticmethod
    def one_task_all_cities_temporal_experiment():
        for city in ["london", "madrid", "melbourne"]:

            obj = ProcessRaw(cityname=city, i_o_length=4,
                             prediction_horizon=1, grid_size=55)

            model = ConvLSTM(
                cityname=city,
                io_length=4,
                pred_horiz=1,
                scale=55,
                shape=(2, 4, 55, 55, 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p001-" + slugify("-temporal-experiment-") + obj.key_dimensions() + ".csv")
            # print (model.model.summary())
            model.train(epochs_param=2, optim="Adam")

    @staticmethod
    def one_task_all_cities_temporal_experiment_bigger_model():
        for city in ["london", "madrid", "melbourne"]:

            obj = ProcessRaw(cityname=city, i_o_length=4,
                             prediction_horizon=1, grid_size=55)

            model = ConvLSTM(
                cityname=city,
                io_length=4,
                pred_horiz=1,
                scale=55,
                shape=(2, 4, 55, 55, 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model_flexible(depth=4, num_filters=128,
                                                              custom_eval=False, BN=True)
            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p001-" + slugify("-temporal-experiment-val-error-big-model") + obj.key_dimensions() + ".csv")
            # print (model.model.summary())
            model.train(epochs_param=15, optim="Adam")
            
    @staticmethod
    def one_task_all_cities_temporal_experiment_bigger_model_use_IC_VAL_DATA():
        for city in ["london", "madrid", "melbourne"]:

            obj = ProcessRaw(cityname=city, i_o_length=4,
                             prediction_horizon=1, grid_size=55)

            model = ConvLSTM(
                cityname=city,
                io_length=4,
                pred_horiz=1,
                scale=55,
                shape=(2, 4, 55, 55, 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model_flexible(depth=4, num_filters=128,
                                                              custom_eval=False, BN=True)
            
            # The easiest way to achieve IC_train data for this function is to change the             
            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p001-" + slugify("-temporal-experiment-val-error-big-model-use-validation-data-for-IC") + obj.key_dimensions() + ".csv")
            # print (model.model.summary())
            model.train(epochs_param=1, optim="Adam")

    @staticmethod
    def one_task_all_cities_temporal_experiment_smaller_model(epochs_param=15):
        for city in ["london", "madrid", "melbourne"]:

            obj = ProcessRaw(cityname=city, i_o_length=4,
                             prediction_horizon=1, grid_size=55)

            model = ConvLSTM(
                cityname=city,
                io_length=4,
                pred_horiz=1,
                scale=55,
                shape=(2, 4, 55, 55, 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model_flexible(depth=2, num_filters=64,
                                                              custom_eval=False, BN=True)
            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p001-" + slugify("-temporal-experiment-val-error-small-model") + obj.key_dimensions() + ".csv")
            # print (model.model.summary())
            model.train(epochs_param=epochs_param, optim="Adam")


    @staticmethod
    def different_tasks_one_model():
        for PRED_HORIZ in range(1, 9):
            obj = ProcessRaw(cityname=config.city_list_def[0], i_o_length=config.i_o_lengths_def[0],
                             prediction_horizon=PRED_HORIZ, grid_size=config.scales_def[0])

            model = ConvLSTM(
                config.city_list_def[0],
                config.i_o_lengths_def[0],
                PRED_HORIZ,
                config.scales_def[0],
                shape=(2, config.i_o_lengths_def[0], config.scales_def[0], config.scales_def[0], 1),
                validation_csv_file=obj.key_dimensions() + "validation.csv",
                log_dir=obj.key_dimensions() + "log_dir",
                custom_eval=False
            )

            model.model = model.create_model()

            model.model = model.create_model_flexible(depth=3, num_filters=64,
                                                              custom_eval=False, BN=False)


            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-default_model-" +
                                                             "-adam-0p1-" + slugify("different_tasks_one_model") + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.model.summary()

            print("======================================")
            os.system("grep \'Trainable params:\' modelsummary.txt")

            # print (model.model.summary())
            model.train(epochs_param=30, optim="Adam")


    @staticmethod
    def one_task_three_models():
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
            custom_eval=False
        )



        # keep only three models
        list_of_models = [
            "create_model_f_def_no_BN",""
            "create_model_small_epochs_no_BN",
            "create_model_f_big_no_BN",
            "create_model_f_shallow_2_no_BN",
            "create_model_f_shallow_1_no_BN",
            "create_model_f_shallow_3_no_BN",
        ]
        # ["create_model_f_def_BN_begin", "create_model_less_filters_BN_begin","create_model_f_shallow_BN_begin", ]
        #  ["create_model_f_big_reg_NO_bn", "create_model_f_def_no_BN", "create_model_small_epochs_no_BN"]

        for model_type in list_of_models:
            updated_model = getattr(model, model_type)()
            model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-" + model_type  + 
                                                     "-adam-0001-" + obj.key_dimensions() + ".csv")
            # print (updated_model.summary())

            from contextlib import redirect_stdout

            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    updated_model.summary()

            print ("======================================")
            sprint (obj.key_dimensions(), model_type)
            os.system("grep \'Trainable params:\' modelsummary.txt")

            print (updated_model.summary())
            model.train(epochs_param=1, optim="Adam")

    @staticmethod
    def print_various_models_summary():
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
            custom_eval=False
        )
        print ("Base model: ")
        print (model.model.summary())
        print ("\n\n\n")
        for filters in [32, 64, 128]:
            for depth in [1, 2, 4]:
                model.model = model.create_model_flexible(depth=depth, num_filters=filters,
                                                          custom_eval=False, BN=True)

                model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-DEP-" + str(depth)
                                                         + "-FIL-" + str(filters) +
                                                         "-adam-01-" + slugify("one_task_different_models") +
                                                         obj.key_dimensions() + ".csv")
                # print (updated_model.summary())

                from contextlib import redirect_stdout

                with open('modelsummary.txt', 'w') as f:
                    with redirect_stdout(f):
                        model.model.summary()

                print("======================================")
                sprint(obj.key_dimensions(), "DEP-" + str(depth) + "-FIL-" + str(filters))
                os.system("grep \'Trainable params:\' modelsummary.txt")

                print (model.model.summary())
                # model.train(epochs_param=30, optim="Adam")

    @staticmethod
    def one_task_increase_lr_midway():
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
            custom_eval=False
        )
        model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-" + "-15-" + ".csv")
        model.train(15)

        model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-" + "-post15-2-" + ".csv")
        optimizer = optimizers.Adam(0.001)
        model.model.compile(optimizer=optimizer, loss="mse", metrics=non_zero_mse)
        model.train(2)
        model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-" + "-post15-5-" + ".csv")
        optimizer = optimizers.Adam(0.001)
        model.model.compile(optimizer=optimizer, loss="mse", metrics=non_zero_mse)
        model.train(3)
        model.validation_csv_file = os.path.join(config.INTERMEDIATE_FOLDER, "validation-" + "-post15-10-" + ".csv")
        optimizer = optimizers.Adam(0.001)
        model.model.compile(optimizer=optimizer, loss="mse", metrics=non_zero_mse)
        model.train(10)

    def create_model_def_without_regularisation(self, custom_eval=False):
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
        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)
        # model.compile(
        #     loss=tensorflow.keras.losses.binary_crossentropy,
        #     optimizer=tensorflow.keras.optimizers.Adam(),
        # )

        return model

    def create_model_shallow_less_filt(self, custom_eval=False):
        _, a, b, c, d = self.shape
        x = np.random.rand(2, a, b, c, d)
        inp = layers.Input(shape=(None, *x.shape[2:]))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = layers.ConvLSTM2D(
            filters=32,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        # x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=8,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

        # Next, we will build the complete model and compile it.
        if not custom_eval:
            model = tensorflow.keras.models.Model(inp, x)
        else:
            model = CustomModel(inp, x)
        # model.compile(
        #     loss=tensorflow.keras.losses.binary_crossentropy,
        #     optimizer=tensorflow.keras.optimizers.Adam(),
        # )
        return model

    def get_methods_of_class(cls):
        return [method for method in dir(cls) if inspect.isfunction(getattr(cls, method))]



if __name__ == "__main__":
    # ConvLSTM.test_ConvLSTM()
    # ConvLSTM.experiment_simple()

    # ConvLSTM.experiment_mix_examples()
    # ConvLSTM.experiment_mix_examples_control()
    # ConvLSTM.experiment_mix_examples_exp_sampling()
    # ConvLSTM.experiment_mix_pred_horiz_2_1()

    # ConvLSTM.print_all_model_summary()
    # ConvLSTM.one_task_different_models()

    # ConvLSTM.one_task_increase_lr_midway()
    # ConvLSTM.one_task_three_models()
    # ConvLSTM.different_tasks_one_model()
    # ConvLSTM.one_task_different_models()
    # ConvLSTM.one_task_one_model_different_BN_no_BN()
    # ConvLSTM.one_task_one_model_with_and_without_lr()
    # ConvLSTM.one_task_all_cities_temporal_experiment()
    # ConvLSTM.one_task_all_cities_temporal_experiment_bigger_model()
    ConvLSTM.one_task_all_cities_temporal_experiment_bigger_model_use_IC_VAL_DATA()
    # ConvLSTM.one_task_all_cities_temporal_experiment_smaller_model(epochs_param=2)
    # ConvLSTM.print_various_models_summary()
