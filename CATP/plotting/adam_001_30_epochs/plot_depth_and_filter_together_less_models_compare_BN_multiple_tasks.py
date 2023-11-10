import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np
from slugify import slugify
from smartprint import smartprint as sprint
import pandas as pd

IO_len = str(4)
SCALE = str(55)
# PRED_HORIZ = str(3)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for PRED_HORIZ in ["1", "2", "3", "4"]:

    dep_colors = [(0.8501191849288735, 0.8501191849288735, 0.8501191849288735, 1.0),
     (0.586082276047674, 0.586082276047674, 0.586082276047674, 1.0),
     (0.3174163783160323, 0.3174163783160323, 0.3174163783160323, 1.0),
     (0.7752402921953095, 0.8583006535947711, 0.9368242983467897, 1.0),
     (0.41708573625528644, 0.6806305267204922, 0.8382314494425221, 1.0),
     (0.1271049596309112, 0.4401845444059977, 0.7074971164936563, 1.0),
    (0.7792233756247597, 0.9132333717800846, 0.7518031526336024, 1.0),
     (0.45176470588235296, 0.7670895809304115, 0.4612072279892349, 1.0),
     (0.1340253748558247, 0.5423298731257208, 0.26828143021914647, 1.0),
     (0.9882352941176471, 0.732072279892349, 0.6299269511726259, 1.0),
     (0.9835755478662053, 0.4127950788158401, 0.28835063437139563, 1.0),
     (0.7925720876585928, 0.09328719723183392, 0.11298731257208766, 1.0),

                  ]


    for List_of_depths, List_of_filters in [
        # ([1], [32, 64, 128]),
        # ([2], [32, 64, 128]),
        # ([3], [32, 64, 128]),
        # ([4], [32, 64, 128]),
        # ([1, 2, 3, 4], [32]),
        # ([1, 2, 3, 4], [64]),
        # ([1, 2, 3, 4], [128]),
        # ([1, 2, 3, 4], [128]),
        # ([2,3,4], [128]),
        # ([1, 2], [32, 64]),
        # [2, 3], [32, 64],
        # [3, 4], [32, 64],
        # [1, 2], [64, 128],
        # [2, 3], [64, 128],
        # [3, 4], [32, 128],
        # ([2, 4], [32, 128]),
        # ([1,2,3,4], [32,64,128]),
        ### ([1,3,4], [32,64,128]),
        # ([3, 4], [32, 128]),
        # ([4], [32, 128]),
        # ([1, 2, 4], [16, 64, 128]),
        ([4], [64]),
        # ([4], [16, 64]),
        # ([3, 4], [128]),
        # ([3, 4], [64]),
        # ([1, 4], [128]),
        # ([3, 4], [32, 64, 128]),
    ]:

        for _, DEP in enumerate(List_of_depths):
            enum_dep = DEP - 1
            for _, FIL in enumerate(List_of_filters):
                enum_fil = [16, 64, 128].index(FIL)
                fname = "validation-default_model--adam-0p001-one-task-one-model-bnlondon-4-"+PRED_HORIZ+"-55-.csv"

                files = {fname: "f_"}

                columns = [
                    'epoch',
                    # 'naïve-model-mse',
                    'val_loss',
                    'MC',
                    # "train_loss",
                ]

                linestyle = {}
                linestyle["val_loss"] = "-"
                linestyle["naïve-model-mse"] = ":"
                linestyle["MC"] = "--"
                linestyle["loss"] = ":"
                linestyle["train_loss"] = ":"

                for idx, file in enumerate(files.keys()):
                    data = pd.read_csv(file)
                    n = 1
                    data["MC"] = data["CSR_MP_sum"]
                    data["IC"] = data["CSR_PM_sum"].max()
                    data["naïve-model-mse"] = data["naive-model-mse"].mean()
                    data["naïve-model-non-zero"] = data["naive-model-non-zero"].mean()
                    data["train_loss"] = data["loss"]
                    data["epoch"] = data["epoch"] + 1 

                    for col in columns:
                        print (col)
                        if col not in ['epoch']:
                            if "train" in col:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]
                            else:
                                alpha_computed = 0.7
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]


                            SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                            data_y = np.convolve(data[col][:], [1 / n] * n, "same")
                            data_y = np.convolve(data_y, [1 / n] * n, "same")

                            # if col == 'MC':
                            #     data_y = data[col].cummax()
                            # elif col == 'val_loss' or col=="loss":
                            #     data_y = data[col].cummin()
                            if col == 'val_loss':
                                # Calculate the cumulative minimum for 'val_loss'
                                data_y = data[col].cummin()

                                # Create a boolean mask where True corresponds to the cumulative minimum
                                is_cummin = data[col] == data_y

                                # Use 'where' to retain the last 'MC' value when the cumulative minimum condition is met
                                data['MC'] = data['MC'].where(is_cummin).ffill()

                            else:
                                data_y = np.convolve(data[col][:], [1 / n] * n, "same")


                            col_label = col
                            if "loss" in col or "mse" in col:
                                axx = ax2
                                # print(data_y)
                                print("Plotted AX2 ")

                            else:
                                axx = ax1
                                # print (data_y)
                                print ("Plotted AX1 ")

                            # Plot the data on the first y-axis

                            # Linewidthcomputed = 2
                            # if "MC" in col:
                            #     Linewidthcomputed = 3

                            alpha_computed = 1
                            if "MC" in col:
                                alpha_computed = 1

                            axx.plot(data['epoch'][:], data_y,
                                    alpha=alpha_computed,
                                    color=color_computed,
                                    # label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                    # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " BN",
                                     label=r"$p_h$: " + PRED_HORIZ + " " + slugify(col_label).replace("mc",
                                                                                                      "MC").replace("val-loss", "Val MSE") + " With BN",                                    # label=f'Simple Label - {col}',
                                    linestyle=linestyle[col],
                                    linewidth=3)

                fname = "validation-default_model--adam-0p001-one-task-one-model-no-bnlondon-4-"+PRED_HORIZ+"-55-.csv"

                files = {fname: "f_"}

                columns = [
                    'epoch',
                    # 'naïve-model-mse',
                    'val_loss',
                    'MC',
                    # "train_loss",
                ]

                linestyle = {}
                linestyle["val_loss"] = "-"
                linestyle["naïve-model-mse"] = ":"
                linestyle["MC"] = "--"
                linestyle["loss"] = ":"
                linestyle["train_loss"] = ":"

                for idx, file in enumerate(files.keys()):
                    data = pd.read_csv(file)
                    n = 1
                    data["MC"] = data["CSR_MP_sum"]
                    data["IC"] = data["CSR_PM_sum"].max()
                    data["naïve-model-mse"] = data["naive-model-mse"].mean()
                    data["naïve-model-non-zero"] = data["naive-model-non-zero"].mean()
                    data["train_loss"] = data["loss"]
                    data["epoch"] = data["epoch"] + 1 

                    for col in columns:
                        if col not in ['epoch']:
                            if "train" in col:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]

                            else:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]



                            SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                            data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            # if col == 'MC':
                            #     data_y = data[col].cummax()
                            # elif col == 'val_loss' or col == "loss":
                            #     data_y = data[col].cummin()
                            # else:
                            #     data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            if col == 'val_loss':
                                # Calculate the cumulative minimum for 'val_loss'
                                data_y = data[col].cummin()

                                # Create a boolean mask where True corresponds to the cumulative minimum
                                is_cummin = data[col] == data_y

                                # Use 'where' to retain the last 'MC' value when the cumulative minimum condition is met
                                data['MC'] = data['MC'].where(is_cummin).ffill()


                            else:
                                data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            col_label = col
                            if "loss" in col or "mse" in col:
                                axx = ax2
                                # print(data_y)
                                print("Plotted AX2 ")

                            else:
                                axx = ax1
                                # print (data_y)
                                print ("Plotted AX1 ")

                            # Plot the data on the first y-axis


                            print (r"$p_h$: " + PRED_HORIZ + " " + slugify( col_label ))
                            axx.plot(data['epoch'][:], data_y,
                                    alpha=0.8,
                                    color=color_computed,
                                    # label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                    # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " No-BN",
                                    # label=f'Simple Label - {col}',
                                    label=r"$p_h$: " + PRED_HORIZ + " " + slugify( col_label ).replace("mc", "MC").replace("val-loss", "Val MSE") + " No-BN",
                                     linestyle=linestyle[col],
                                    linewidth=1.5)



plt.title('Effect of Batch Normalisation\n Model used: ' +  "f-DEP-" + str(DEP) + "-FIL-" +
          str(FIL), fontsize=8)
# plt.xlabel('Epoch')
# ax1.set_yticks(range(0, 1000, 100))
# ax2.set_yticks(range(0, 1000, 100))
ax1.set_xlabel('Epochs', fontsize=11)
ax2.set_ylabel('Val MSE', color='black', fontsize=11)
ax1.set_ylabel('MC', color='black', fontsize=11)
# ax1.set_ylim(0, 700)
# ax2.set_ylim(0, 2000)
ax2.set_yscale("log")
ax1.set_yscale("log")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines2 + lines
all_labels = labels2 + labels
ax1.legend(all_lines, all_labels, loc='center right', ncol=2, fontsize=7.5)

# plt.legend()

# plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=6)
ax1.set_xticks(list(range(0, 30, 2)))#, fontsize=11)
# plt.grid(axis='x', alpha=0.05)
# plt.ylim(0, 1000)
plt.tight_layout()

plt.savefig("london-IO_LEN" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + \
            "_d_".join([str(x) for x in List_of_depths]) + \
            "_f_".join([str(x) for x in List_of_filters]) \
         +"_ph_".join(["1234"])  \
            +"_BN_compare_multiple_Tasks__all_4_without_training_loss_selected_epochs_where_val_decrease.png", dpi=300)
plt.show()









import numpy as np
from scipy import fftpack
from statsmodels.tsa.stattools import acf




fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for PRED_HORIZ in ["1", "2", "3", "4"]:

    dep_colors = [(0.8501191849288735, 0.8501191849288735, 0.8501191849288735, 1.0),
     (0.586082276047674, 0.586082276047674, 0.586082276047674, 1.0),
     (0.3174163783160323, 0.3174163783160323, 0.3174163783160323, 1.0),
     (0.7752402921953095, 0.8583006535947711, 0.9368242983467897, 1.0),
     (0.41708573625528644, 0.6806305267204922, 0.8382314494425221, 1.0),
     (0.1271049596309112, 0.4401845444059977, 0.7074971164936563, 1.0),
    (0.7792233756247597, 0.9132333717800846, 0.7518031526336024, 1.0),
     (0.45176470588235296, 0.7670895809304115, 0.4612072279892349, 1.0),
     (0.1340253748558247, 0.5423298731257208, 0.26828143021914647, 1.0),
     (0.9882352941176471, 0.732072279892349, 0.6299269511726259, 1.0),
     (0.9835755478662053, 0.4127950788158401, 0.28835063437139563, 1.0),
     (0.7925720876585928, 0.09328719723183392, 0.11298731257208766, 1.0),

                  ]


    for List_of_depths, List_of_filters in [
        # ([1], [32, 64, 128]),
        # ([2], [32, 64, 128]),
        # ([3], [32, 64, 128]),
        # ([4], [32, 64, 128]),
        # ([1, 2, 3, 4], [32]),
        # ([1, 2, 3, 4], [64]),
        # ([1, 2, 3, 4], [128]),
        # ([1, 2, 3, 4], [128]),
        # ([2,3,4], [128]),
        # ([1, 2], [32, 64]),
        # [2, 3], [32, 64],
        # [3, 4], [32, 64],
        # [1, 2], [64, 128],
        # [2, 3], [64, 128],
        # [3, 4], [32, 128],
        # ([2, 4], [32, 128]),
        # ([1,2,3,4], [32,64,128]),
        ### ([1,3,4], [32,64,128]),
        # ([3, 4], [32, 128]),
        # ([4], [32, 128]),
        # ([1, 2, 4], [16, 64, 128]),
        ([4], [64]),
        # ([4], [16, 64]),
        # ([3, 4], [128]),
        # ([3, 4], [64]),
        # ([1, 4], [128]),
        # ([3, 4], [32, 64, 128]),
    ]:

        for _, DEP in enumerate(List_of_depths):
            enum_dep = DEP - 1
            for _, FIL in enumerate(List_of_filters):
                enum_fil = [16, 64, 128].index(FIL)
                fname = "validation-default_model--adam-0p001-one-task-one-model-bnlondon-4-"+PRED_HORIZ+"-55-.csv"

                files = {fname: "f_"}

                columns = [
                    'epoch',
                    # 'naïve-model-mse',
                    'val_loss',
                    'MC',
                    # "train_loss",
                ]

                linestyle = {}
                linestyle["val_loss"] = "-"
                linestyle["naïve-model-mse"] = ":"
                linestyle["MC"] = "--"
                linestyle["loss"] = ":"
                linestyle["train_loss"] = ":"

                for idx, file in enumerate(files.keys()):
                    data = pd.read_csv(file)
                    n = 1
                    data["MC"] = data["CSR_MP_sum"]
                    data["IC"] = data["CSR_PM_sum"].max()
                    data["naïve-model-mse"] = data["naive-model-mse"].mean()
                    data["naïve-model-non-zero"] = data["naive-model-non-zero"].mean()
                    data["train_loss"] = data["loss"]
                    data["epoch"] = data["epoch"] + 1 

                    for col in columns:
                        print (col)
                        if col not in ['epoch']:
                            if "train" in col:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]
                            else:
                                alpha_computed = 0.7
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]


                            SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                            data_y = np.convolve(data[col][:], [1 / n] * n, "same")
                            data_y = np.convolve(data_y, [1 / n] * n, "same")

                            if col == 'val_loss':
                                # Calculate the cumulative minimum for 'val_loss'
                                data_y = data[col].cummin()

                                # Create a boolean mask where True corresponds to the cumulative minimum
                                is_cummin = data[col] == data_y

                                # Use 'where' to retain the last 'MC' value when the cumulative minimum condition is met
                                data['MC'] = data['MC'].where(is_cummin).ffill()


                            else:
                                data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            # data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            val_mse_model_BN = data["MC"]


                            col_label = col
                            if "loss" in col or "mse" in col:
                                axx = ax2
                                # print(data_y)
                                print("Plotted AX2 ")

                            else:
                                axx = ax1
                                # print (data_y)
                                print ("Plotted AX1 ")

                            # Plot the data on the first y-axis

                            # Linewidthcomputed = 2
                            # if "MC" in col:
                            #     Linewidthcomputed = 3

                            alpha_computed = 1
                            if "MC" in col:
                                alpha_computed = 1

                            axx.plot(data['epoch'][:], data_y,
                                    alpha=alpha_computed,
                                    color=color_computed,
                                    # label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                    # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " BN",
                                     label=r"$p_h$: " + PRED_HORIZ + " " + slugify(col_label).replace("mc",
                                                                                                      "MC").replace("val-loss", "Val MSE") + " With BN",                                    # label=f'Simple Label - {col}',
                                    linestyle=linestyle[col],
                                    linewidth=3)

                fname = "validation-default_model--adam-0p001-one-task-one-model-no-bnlondon-4-"+PRED_HORIZ+"-55-.csv"

                files = {fname: "f_"}

                columns = [
                    'epoch',
                    # 'naïve-model-mse',
                    'val_loss',
                    'MC',
                    # "train_loss",
                ]

                linestyle = {}
                linestyle["val_loss"] = "-"
                linestyle["naïve-model-mse"] = ":"
                linestyle["MC"] = "--"
                linestyle["loss"] = ":"
                linestyle["train_loss"] = ":"

                for idx, file in enumerate(files.keys()):
                    data = pd.read_csv(file)
                    n = 1
                    data["MC"] = data["CSR_MP_sum"]
                    data["IC"] = data["CSR_PM_sum"].max()
                    data["naïve-model-mse"] = data["naive-model-mse"].mean()
                    data["naïve-model-non-zero"] = data["naive-model-non-zero"].mean()
                    data["train_loss"] = data["loss"]
                    data["epoch"] = data["epoch"] + 1 

                    for col in columns:
                        if col not in ['epoch']:
                            if "train" in col:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]

                            else:
                                alpha_computed = 1
                                color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]
                                color_computed = ["#01796F", "#FFD700", "#DC143C", "#32CD32"][int(PRED_HORIZ)-1]



                            SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                            data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            if col == 'val_loss':
                                # Calculate the cumulative minimum for 'val_loss'
                                data_y = data[col].cummin()

                                # Create a boolean mask where True corresponds to the cumulative minimum
                                is_cummin = data[col] == data_y

                                # Use 'where' to retain the last 'MC' value when the cumulative minimum condition is met
                                data['MC'] = data['MC'].where(is_cummin).ffill()


                            else:
                                data_y = np.convolve(data[col][:], [1 / n] * n, "same")


                            # data_y = np.convolve(data[col][:], [1 / n] * n, "same")

                            col_label = col
                            if "loss" in col or "mse" in col:
                                axx = ax2
                                # print(data_y)
                                print("Plotted AX2 ")

                            else:
                                axx = ax1
                                # print (data_y)
                                print ("Plotted AX1 ")

                            # Plot the data on the first y-axis


                            print (r"$p_h$: " + PRED_HORIZ + " " + slugify( col_label ))
                            axx.plot(data['epoch'][:], data_y,
                                    alpha=0.8,
                                    color=color_computed,
                                    # label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                    # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " No-BN",
                                    # label=f'Simple Label - {col}',
                                    label=r"$p_h$: " + PRED_HORIZ + " " + slugify( col_label ).replace("mc", "MC").replace("val-loss", "Val MSE") + " No-BN",
                                     linestyle=linestyle[col],
                                    linewidth=1.5)
                            val_mse_model_no_BN = data["MC"]

                            # Point 3: Signal-to-Noise Ratio (SNR)
                            # This is an adaptation of SNR from signal processing for validation MSE.
                            def calculate_snr(mse_values, epsilon=1e-10):
                                signal_power = np.mean(mse_values) ** 2
                                noise_power = np.var(mse_values) + epsilon  # Added epsilon to avoid division by zero
                                return 10 * np.log10(signal_power / noise_power)


                            snr_model1 = calculate_snr(val_mse_model_BN)
                            snr_model2 = calculate_snr(val_mse_model_no_BN)


                            # Point 4: Loss Stability Score
                            def loss_stability_score(mse_values):
                                return np.mean(mse_values) / (
                                            np.std(mse_values) + 1e-11)


                            val_mse_model1 = val_mse_model_BN
                            val_mse_model2 = val_mse_model_no_BN

                            stability_score_model1 = loss_stability_score(val_mse_model1)
                            stability_score_model2 = loss_stability_score(val_mse_model2)

                            # Point 6: Autocorrelation
                            # We use the autocorrelation function (acf) from the statsmodels library.
                            # This will give us the correlation of the series with itself at different lags.
                            autocorrelation_model1 = np.mean(acf(val_mse_model1))
                            autocorrelation_model2 = np.mean(acf(val_mse_model2))

                            # Print results
                            print(f"SNR Model 1: {snr_model1}")
                            print(f"SNR Model 2: {snr_model2}")
                            print(f"Stability Score Model 1: {stability_score_model1}")
                            print(f"Stability Score Model 2: {stability_score_model2}")
                            print(f"Autocorrelation Model 1: {autocorrelation_model1}")
                            print(f"Autocorrelation Model 2: {autocorrelation_model2}")

plt.title('Effect of Batch Normalisation\n Model used: ' +  "f-DEP-" + str(DEP) + "-FIL-" +
          str(FIL), fontsize=8)
# plt.xlabel('Epoch')
# ax1.set_yticks(range(0, 1000, 100))
# ax2.set_yticks(range(0, 1000, 100))
ax1.set_xlabel('Epochs', fontsize=11)
ax2.set_ylabel('Val MSE', color='black', fontsize=11)
ax1.set_ylabel('MC', color='black', fontsize=11)
# ax1.set_ylim(0, 700)
# ax2.set_ylim(0, 2000)
ax2.set_yscale("log")
ax1.set_yscale("log")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines2 + lines
all_labels = labels2 + labels
ax1.legend(all_lines, all_labels, loc='center right', ncol=2, fontsize=7.5)

# plt.legend()

# plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=6)
ax1.set_xticks(list(range(0, 30, 2)))#, fontsize=11)
# plt.grid(axis='x', alpha=0.05)
# plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig("BN_plot.png", dpi=300)
# plt.savefig("london-IO_LEN" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + \
#             "_d_".join([str(x) for x in List_of_depths]) + \
#             "_f_".join([str(x) for x in List_of_filters]) \
#          +"_ph_".join(["1234"])  \
#             +"_BN_compare_multiple_Tasks__all_4_without_training_loss_all_epochs.png", dpi=300)
plt.show()
