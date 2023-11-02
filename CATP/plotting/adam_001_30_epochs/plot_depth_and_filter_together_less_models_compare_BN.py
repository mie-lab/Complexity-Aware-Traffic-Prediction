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
PRED_HORIZ = str(1)


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
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    NAIVE_BASELINE_PLOTTED = False
    for _, DEP in enumerate(List_of_depths):
        enum_dep = DEP - 1
        for _, FIL in enumerate(List_of_filters):
            enum_fil = [16, 64, 128].index(FIL)
            fname = "validation-default_model--adam-0p1-one-task-one-model-bnlondon-4-1-55-.csv"

            files = {fname: "f_"}

            columns = [
                'epoch',
                'naïve-model-mse',
                'val_loss',
                'MC',
                "train_loss",
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

                for col in columns:
                    if NAIVE_BASELINE_PLOTTED and "naïve" in col:
                        continue
                    elif not NAIVE_BASELINE_PLOTTED and "naïve" in col:
                        NAIVE_BASELINE_PLOTTED = True

                    if col not in ['epoch']:
                        if "naïve" in col:
                            alpha_computed = 1
                            color_computed = "black"
                        else:
                            alpha_computed = 0.7
                            color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]

                        SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                        data_y = np.convolve(data[col][:], [1 / n] * n, "same")
                        data_y = np.convolve(data_y, [1 / n] * n, "same")

                        col_label = col
                        if "loss" in col or "mse" in col:
                            axx = ax2
                            print(data_y)
                            print("Plotted AX2 ")

                        else:
                            axx = ax1
                            print (data_y)
                            print ("Plotted AX1 ")

                        # Plot the data on the first y-axis


                        axx.plot(data['epoch'][:], data_y,
                                alpha=0.5,
                                color=color_computed,
                                label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " BN",
                                # label=f'Simple Label - {col}',
                                linestyle=linestyle[col],
                                linewidth=3)

            fname = "validation-default_model--adam-0p1-one-task-one-model-no-bnlondon-4-1-55-.csv"

            files = {fname: "f_"}

            columns = [
                'epoch',
                'naïve-model-mse',
                'val_loss',
                'MC',
                "train_loss",
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

                for col in columns:
                    if NAIVE_BASELINE_PLOTTED and "naïve" in col:
                        continue
                    elif not NAIVE_BASELINE_PLOTTED and "naïve" in col:
                        NAIVE_BASELINE_PLOTTED = True

                    if col not in ['epoch']:
                        if "naïve" in col:
                            alpha_computed = 1
                            color_computed = "black"
                        else:
                            alpha_computed = 0.7
                            color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]

                        SCALING_LOSS_FACTOR = 1  # Set scaling factor to 1
                        data_y = np.convolve(data[col][:], [1 / n] * n, "same")
                        data_y = np.convolve(data_y, [1 / n] * n, "same")

                        col_label = col
                        if "loss" in col or "mse" in col:
                            axx = ax2
                            print(data_y)
                            print("Plotted AX2 ")

                        else:
                            axx = ax1
                            print (data_y)
                            print ("Plotted AX1 ")

                        # Plot the data on the first y-axis


                        axx.plot(data['epoch'][:], data_y,
                                alpha=1,
                                color=color_computed,
                                label=(col_label + slugify( "_" + str(files[file])).replace("mc", " MC ")
                                + "-DEP-" + str(DEP) + "-FIL-" + str(FIL)).replace("MCf", "MC-f").replace("loss-f", "MSE-f").replace("lossf","loss f").replace("sf","s f").replace("_", " ") + " No-BN",
                                # label=f'Simple Label - {col}',
                                linestyle=linestyle[col],
                                linewidth=3)



    plt.title('Evaluation of Model Complexity during training', fontsize=8)
    # plt.xlabel('Epoch')
    # ax1.set_yticks(range(0, 1000, 100))
    # ax2.set_yticks(range(0, 1000, 100))
    ax1.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Losses', color='black', fontsize=11)
    ax1.set_ylabel('MC', color='black', fontsize=11)
    # ax1.set_ylim(0, 700)
    # ax2.set_ylim(0, 2000)
    ax2.set_yscale("log")


    plt.legend()

    # plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=6)
    ax1.set_xticks(list(range(0, 20, 2)))#, fontsize=11)
    # plt.grid(axis='x', alpha=0.05)
    # plt.ylim(0, 1000)
    plt.tight_layout()

    plt.savefig("london-IO_LEN" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + \
                "_d_".join([str(x) for x in List_of_depths]) + \
                "_f_".join([str(x) for x in List_of_filters]) + \
             "_BN_compare.png", dpi=300)
    plt.show()
