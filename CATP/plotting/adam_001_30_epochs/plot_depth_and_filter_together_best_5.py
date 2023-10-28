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
    ([1, 2, 4], [16, 64, 128]),
    # ([3, 4], [128]),
    # ([3, 4], [64]),
    # ([1, 4], [128]),
    # ([3, 4], [32, 64, 128]),
]:
    plt.clf()
    NAIVE_BASELINE_PLOTTED = False
    for _, DEP in enumerate(List_of_depths):
        enum_dep = DEP - 1
        for _, FIL in enumerate(List_of_filters):
            enum_fil = [16, 64, 128].index(FIL)
            fname = f"validation-DEP-{DEP}-FIL-{FIL}-adam-01-one-task-different-modelslondon-{IO_len}-{PRED_HORIZ}-{SCALE}-.csv"
            files = {fname: "f_"}

            columns = [
                'epoch',
                'naive-model-mse',
                'val_loss',
                'MC',
                # "loss",
            ]

            linestyle = {}
            linestyle["val_loss"] = "-"
            linestyle["naive-model-mse"] = ":"
            linestyle["MC"] = "--"
            linestyle["loss"] = ":"

            for idx, file in enumerate(files.keys()):
                data = pd.read_csv(file)
                n = 1
                data["MC"] = data["CSR_MP_sum"]
                data["IC"] = data["CSR_PM_sum"].max()
                data["naive-model-mse"] = data["naive-model-mse"].mean()
                data["naive-model-non-zero"] = data["naive-model-non-zero"].mean()

                for col in columns:
                    if NAIVE_BASELINE_PLOTTED and "naive" in col:
                        continue
                    elif not NAIVE_BASELINE_PLOTTED and "naive" in col:
                        NAIVE_BASELINE_PLOTTED = True

                    if col not in ['epoch']:
                        if "naive" in col:
                            alpha_computed = 1
                            color_computed = "black"
                        else:
                            alpha_computed = 0.7
                            color_computed = dep_colors[enum_dep * len([1,2,3]) + enum_fil ]

                        SCALING_LOSS_FACTOR = 3
                        data_y = np.convolve(data[col][:], [1 / n] * n, "same")
                        data_y = np.convolve(data_y, [1 / n] * n, "same")

                        if col in ["val_loss", "naive-model-mse"]:
                            data_y = data_y/ SCALING_LOSS_FACTOR
                            col_label = col + " * " + str(SCALING_LOSS_FACTOR) + " "
                        else:
                            col_label = col
                        if SCALING_LOSS_FACTOR == 1:
                            col_label = col

                        plt.plot(data['epoch'][:], data_y,
                                     alpha=alpha_computed,
                                     color=color_computed,
                                     label=col_label + slugify( "_" + str(files[file])).replace("mc-", "MC-")
                                     + "-DEP-" + str(DEP) + "-FIL-" + str(FIL),
                                     linestyle=linestyle[col])



    plt.title('Evaluation of Model Complexity during training', fontsize=8)
    plt.xlabel('Epoch')
    # plt.ylabel('Value')
    plt.legend(fontsize=6, ncol=3, loc="best")
    plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=6)
    plt.grid(axis='x', alpha=0.05)
    plt.ylim(0, 900)
    # plt.xlim(n, 30-4)
    # plt.yscale("log")
    plt.tight_layout()

    # plt.savefig("london-IO_LEN" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + \
    #             "_d_".join([str(x) for x in List_of_depths]) + \
    #             "_f_".join([str(x) for x in List_of_filters]) + \
    #          ".png", dpi=300)
    plt.show()

    plt.clf()

    line_plot_x = {}
    line_plot_y = {}

    for _, DEP in enumerate(List_of_depths):
        enum_dep = DEP - 1



        for _, FIL in enumerate(List_of_filters):
            enum_fil = [16, 64, 128].index(FIL)
            fname = f"validation-DEP-{DEP}-FIL-{FIL}-adam-01-one-task-different-modelslondon-{IO_len}-{PRED_HORIZ}-{SCALE}-.csv"
            files = {fname: "f_"}

            columns = [
                'epoch',
                'naive-model-mse',
                'val_loss',
                'MC'
            ]

            linestyle = {}
            linestyle["val_loss"] = "-"
            linestyle["naive-model-mse"] = ":"
            linestyle["MC"] = "-."
            SIZE_SCATTER = 2

            for idx, file in enumerate(files.keys()):
                data = pd.read_csv(file)
                n = 1
                data["MC"] = data["CSR_MP_sum"]
                data["IC"] = data["CSR_PM_sum"].max()
                data["naive-model-mse"] = data["naive-model-mse"].mean()
                data["naive-model-non-zero"] = data["naive-model-non-zero"].mean()

                val_loss = data["val_loss"]
                argmin = (val_loss).argsort()[:5]
                #
                # argmin = np.argmax(700 - data["val_loss"])
                # argmin = np.where(data["val_loss"] < data["naive-model-mse"])

                # plt.scatter(data['MC'][argmin], data["val_loss"][argmin],
                # plt.scatter(np.median(data['MC'].to_numpy()[argmin].flatten()),
                #             np.median(data["val_loss"].to_numpy()[argmin].flatten()),

                # plt.scatter(data['MC'][16:].median(), data["val_loss"][16:].median(),

                plt.scatter(data['MC'][argmin].mean(), data["val_loss"][argmin].mean(),
                        alpha=1,
                            color=dep_colors[enum_dep * len([1,2,3]) + enum_fil ],
                            label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                         + "-DEP-" + str(DEP) + "-FIL-" + str(FIL) + " Min",
                         # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL),
                            )

                line_plot_x[DEP, FIL] = data["MC"][argmin].mean()
                line_plot_y[DEP, FIL] = data["val_loss"][argmin].mean()

                # line_plot_x[DEP, FIL] = np.median(data["MC"].to_numpy()[argmin].flatten())  # [16:].median()
                # line_plot_y[DEP, FIL] = np.median(data["val_loss"].to_numpy()[argmin].flatten())  # [16:].median()

                # argmax = np.argmax(700 - data["val_loss"])
                # plt.scatter(data['MC'][argmax], data["val_loss"][argmax],
                # plt.scatter(data['MC'][:argmax], data["val_loss"][:argmax],
                #         alpha=1, #data["epoch"]/20,
                #             color=dep_colors[enum_dep * len([1,2,3]) + enum_fil ],
                            # label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                         # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL),
                         #    s=SIZE_SCATTER,
                         #    )

                # plt.scatter(data['MC'][data.shape[0]-1], data["val_loss"][data.shape[0]-1],
                #         alpha=1,
                #             color=dep_colors[enum_dep * len([1,2,3]) + enum_fil ],
                #             label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                #          + "-DEP-" + str(DEP) + "-FIL-" + str(FIL),
                #             # s=SIZE_SCATTER
                #             )

                # plt.scatter(data['MC'][20:].median(), data["val_loss"][20:].median(),
                #         alpha=1,
                #             color=dep_colors[enum_dep * len([1,2,3]) + enum_fil ],
                #             label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                #          + "-DEP-" + str(DEP) + "-FIL-" + str(FIL) + " Median",
                #             # s=SIZE_SCATTER
                #             )



        plt.plot([line_plot_x[DEP, 16],  line_plot_x[DEP, 64], line_plot_x[DEP, 128]],
                [line_plot_y[DEP, 16],  line_plot_y[DEP, 64], line_plot_y[DEP, 128]],
                 color="black",
                 alpha=0.2)
    # plt.plot([line_plot_x[1, 16], line_plot_x[2, 16], line_plot_x[4, 16]],
    #         [line_plot_y[1, 16], line_plot_y[2, 16], line_plot_y[4, 16]],
    #          color="blue",
    #          alpha=0.1)
    # plt.plot([line_plot_x[1, 64], line_plot_x[2, 64], line_plot_x[4, 64]],
    #          [line_plot_y[1, 64], line_plot_y[2, 64], line_plot_y[4, 64]],
    #          color="blue",
    #          alpha=0.4)
    # plt.plot([line_plot_x[1, 128], line_plot_x[2, 128], line_plot_x[4, 128]],
    #          [line_plot_y[1, 128], line_plot_y[2, 128], line_plot_y[4, 128]],
    #          color="blue",
    #          alpha=0.9)

    plt.title('Val MSE vs MC', fontsize=8)
    plt.xlabel('MC')
    plt.ylabel('Val MSE')
    plt.legend(fontsize=7, ncol=3, loc="upper right")
    # plt.xlim(0, 750)
    # plt.ylim(550, 850)
    # plt.ylim(0, 2000)
    # plt.xscale("symlog")
    # plt.yscale("symlog")
    plt.tight_layout()
    plt.savefig("london-IO_LEN_scatter_all_combined_" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE +\
                "_d_".join([str(x) for x in List_of_depths]) +\
                "_f_".join([str(x) for x in List_of_filters]) +\
                "best_5_mean.png", dpi=300)
    sprint ("_d_".join([str(x) for x in List_of_depths]), List_of_depths)

