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
    ([1, 2, 4], [16, 64, 128]),
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
                            color_computed = dep_colors[enum_dep * len([1, 2, 4]) + enum_fil ]

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
    plt.legend(fontsize=6, ncol=3, loc="best")
    plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=6)
    plt.grid(axis='x', alpha=0.05)
    plt.ylim(0, 900)
    plt.tight_layout()

    plt.savefig("london-IO_LEN" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + \
                "_d_".join([str(x) for x in List_of_depths]) + \
                "_f_".join([str(x) for x in List_of_filters]) + \
             "_less_4_models.png", dpi=300)
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

                argmin = np.argmin(data["val_loss"])
                plt.scatter(data['MC'][argmin], data["val_loss"][argmin],
                        alpha=1,
                            color=dep_colors[enum_dep * len([1, 2, 4]) + enum_fil ],
                            label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                         + "-DEP-" + str(DEP) + "-FIL-" + str(FIL) + " Min",
                         # + "-DEP-" + str(DEP) + "-FIL-" + str(FIL),
                            )

                line_plot_x[DEP, FIL] = data["MC"][argmin]
                line_plot_y[DEP, FIL] = data["val_loss"][argmin]



    plt.title('Val MSE vs MC', fontsize=8)
    plt.xlabel('MC')
    plt.ylabel('Val MSE')
    plt.legend(fontsize=7, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.savefig("london-IO_LEN_scatter_all_combined_" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE +\
                "_d_".join([str(x) for x in List_of_depths]) +\
                "_f_".join([str(x) for x in List_of_filters]) +\
                "min_fil_less_4_models.png", dpi=300)
    sprint ("_d_".join([str(x) for x in List_of_depths]), List_of_depths)


    plt.clf()
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms


    def confidence_ellipse(x, y, ax, n_std=1, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

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
                'loss'
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

                # argmin = np.argmin(data["val_loss"])
                val_loss = data["val_loss"]
                argmin = (val_loss).argsort()[:5]
                plt.scatter(data['MC'][argmin], data["val_loss"][argmin],
                        alpha=1,
                            color=dep_colors[enum_dep * len([1, 2, 4]) + enum_fil ],
                            label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                         + "-DEP-" + str(DEP) + "-FIL-" + str(FIL) ,
                            )
                if enum_fil == 2:
                    plt.scatter(data['MC'][argmin], data["loss"][argmin],
                            alpha=1,
                                color=dep_colors[enum_dep * len([1, 2, 4]) + enum_fil ],
                                label=slugify(col + "_" + str(files[file])).replace("mc-", "MC-")
                             + "-DEP-" + str(DEP) + "-FIL-" + str(FIL) ,
                                marker="*"
                                )
                else:
                    plt.scatter(data['MC'][argmin], data["loss"][argmin],
                                alpha=1,
                                color=dep_colors[enum_dep * len([1, 2, 4]) + enum_fil],
                                marker="*"
                                )
                # confidence_ellipse(data['MC'][:], data["val_loss"][:], plt.gca(),
                #                    edgecolor=dep_colors[enum_dep * len([1, 2, 4]) + enum_fil])

                sprint (DEP, FIL, (data['MC'][argmin]).mean() )
    plt.title('Train and Val MSE vs MC', fontsize=8)
    plt.xlabel('MC')
    plt.ylabel('MSE')
    legend = plt.legend(fontsize=8.5, loc="best", ncol=3, edgecolor="black")
    legend.get_frame().set_alpha(0.3)
    # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    # plt.tight_layout()
    plt.savefig("london-IO_LEN_scatter_all_combined_" + IO_len + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE +\
                "_d_".join([str(x) for x in List_of_depths]) +\
                "_f_".join([str(x) for x in List_of_filters]) +\
                "-all-_min_less_4_models.png", dpi=300)
    sprint ("_d_".join([str(x) for x in List_of_depths]), List_of_depths)




