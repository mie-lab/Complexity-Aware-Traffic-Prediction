import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np
from slugify import slugify

IO_len = str(4)
SCALE = str(55)
PRED_HORIZ = str(6)
# Define the CSV filenames
files = {
    "validation-create_model_f_def_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_deep_medium_filters",
    "validation-create_model_f_big_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_deep_high_filters",
    "validation-create_model_f_shallow_1_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_shallow_1",
    "validation-create_model_f_shallow_2_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_shallow_2",
    "validation-create_model_f_shallow_3_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_shallow_3",
    "validation-create_model_small_epochs_no_BN-adam-0001-london-" + IO_len + "-" + PRED_HORIZ + "-" + SCALE + "-.csv": "f_deep_low_filters"
}


colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]  # these are 10 distinguishable colors

# Create a dictionary mapping each file to a color
colors_for_file = {file: colors[i % len(colors)] for i, file in enumerate(files)}

alphas = [0.1, 0.2,  0.4, 0.8, 0.9, 1] # , 0.1, 0.5]
# alphas = np.array([1,2,3,4,5,6])/6
# alphas = [0.3, 1, 0.2]
columns = [
            'epoch',
             # 'CSR_GB_sum',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             # 'CSR_MP_sum',
             # 'CSR_NM_sum',
             # 'CSR_PM_std',
             # 'CSR_PM_sum',
             # 'loss',
             'naive-model-mse',
             # 'naive-model-non-zero',
             # 'non_zero_mse',
             # 'self.CSR_GB_count',
             # 'self.CSR_NM_count',
             # 'self.CSR_PM_count',
             'val_loss',
             # 'val_non_zero_mse',
                "MC",
                # "IC"
            ]




color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
color_dict["IC"] = 'green'
color_dict["MC"] = 'red'
color_dict['val_loss'] = 'blue'
color_dict['val_non_zero_mse'] = 'blue'
color_dict['loss'] = 'orange'
color_dict['naive-model-mse'] = 'black'
color_dict['naive-model-non-zero'] = 'black'

# Loop through the CSV files
for idx, file in enumerate(files.keys()):

    # Load the data
    data = pd.read_csv(file)

    # Plot each column on the same plot
    n = 1

    # data["val/train"] = data["val_non_zero_mse"] / data["non_zero_mse"]
    data["MC"] = data["CSR_MP_sum"]
    data["IC"] = data["CSR_PM_sum"].max()
    data["naive-model-mse"] = data["naive-model-mse"].mean()
    data["naive-model-non-zero"] = data["naive-model-non-zero"].mean()

    linestyle = '-'
    # if "create_model_f_big" in file:
    #     linestyle = ":"
    for col in columns:

        if idx >= 0 :
            # Exclude 'epoch' column from plotting
            if idx > 0 and "naive" in col:
                continue

            if col not in ['epoch']: #, "MC", "IC"]:

                # max_ = data[col].max()
                # data[col] = data[col] / max_
                alpha_computed = alphas[idx]
                if "naive" in col:
                    alpha_computed = 1
                plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                         alpha=alpha_computed,
                         color=color_dict[col],
                         label=slugify( col + "_" + str(files[file])),
                         linestyle=linestyle)



plt.title('MC evaluation during training', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
# plt.legend(fontsize=6, ncol=2, loc="upper right")
plt.legend(fontsize=8, ncol=2, loc="best")
plt.xticks(list(range(0, 20, 1)), rotation=90, fontsize=4)
# plt.yscale("log")

plt.grid(axis='x',alpha=0.05)
# plt.xlim(3//2, 30-n)
# plt.xlim(0, 10)
# plt.ylim(1, 3000)
plt.tight_layout()

plt.savefig("london-IO_LEN" + IO_len  + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + ".png", dpi=300)
plt.show()

alphas = [1] * 100

plt.clf()
# Loop through the CSV files
for idx, file in enumerate(files.keys()):

    # Load the data
    data = pd.read_csv(file)

    # Plot each column on the same plot
    n = 1

    # data["val/train"] = data["val_non_zero_mse"] / data["non_zero_mse"]
    data["MC"] = data["CSR_MP_sum"]
    data["IC"] = data["CSR_PM_sum"].max()
    data["naive-model-mse"] = data["naive-model-mse"].mean()
    data["naive-model-non-zero"] = data["naive-model-non-zero"].mean()

    linestyle = '-'
    # if "create_model_f_big" in file:
    #     linestyle = ":"
    for col in columns[:1]:

        if idx >= 0 :
            # Exclude 'epoch' column from plotting

            if col in ["epoch"]: # ["IC"]

                scale = 1

                if col in ["MC", "IC"]:
                    data[col] = data[col] * scale

                elif col in ["val_loss", "loss", "val_non_zero_mse", "non_zero_mse",
                             "naive-model-non-zero", "naive-model-mse"]:
                    data[col] = (data[col] * scale * scale)
                    # continue
                # max_ = data[col].max()
                # data[col] = data[col] / max_

                plt.scatter(data['MC'], data["val_loss"],
                         alpha=0.7, #(data["epoch"]/np.max(data["epoch"])).tolist(),
                            color=[colors_for_file[file]] * data["MC"].shape[0],
                            edgecolors=(1,1,1,0),
                            linewidths=0.1,
                            label=slugify(str(files[file])),
                            # s=11
                            )
                # plt.scatter(np.mean(data['MC']) , np.mean(data["val_loss"]),
                #             label=slugify(str(files[file])) + " mean",
                #             marker="*",
                #             color=colors_for_file[file],
                #             alpha=1,
                #             s=70)
                # plt.scatter(data['IC'].mean() - np.mean(data['MC'][10:]) , np.min(data["val_loss"]),
                #             label=slugify(str(files[file])) + " mean",
                #             marker="*",
                #             color=colors_for_file[file],
                #             alpha=1,
                #             s=70)
                # if idx == 0 :
                #     plt.scatter([data['IC'].mean()] * (1450-1150) , list (range(1150, 1450)),
                #                 label=" IC",
                #                 marker="o",
                #                 color=colors_for_file[file],
                #                 alpha=1,
                #                 s=1)



plt.legend(fontsize=10, loc="upper right")
# plt.yscale("log")
# plt.xscale("log")
# plt.gca().set_aspect(0.003)
# plt.ylim(1150, 1450)
# plt.xlim(0, 130)
plt.title("Val loss vs Model Complexity")
plt.ylabel("Validation loss (MSE)")
plt.xlabel("Model Complexity (MC|f) at various epochs")
plt.tight_layout()
plt.savefig("london_scatter_Val_vs_MC-IO_LEN" + IO_len  + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + ".png", dpi=300)



alphas = [1] * 100

plt.clf()
# Loop through the CSV files
for idx, file in enumerate(files.keys()):

    # Load the data
    data = pd.read_csv(file)

    # Plot each column on the same plot
    n = 1

    # data["val/train"] = data["val_non_zero_mse"] / data["non_zero_mse"]
    data["MC"] = data["CSR_MP_sum"]
    data["IC"] = data["CSR_PM_sum"].max()
    data["naive-model-mse"] = data["naive-model-mse"].mean()
    data["naive-model-non-zero"] = data["naive-model-non-zero"].mean()

    linestyle = '-'
    # if "create_model_f_big" in file:
    #     linestyle = ":"
    for col in columns[:1]:

        if idx >= 0 :
            # Exclude 'epoch' column from plotting

            if col in ["epoch"]: # ["IC"]


                scale = 1

                if col in ["MC", "IC"]:
                    data[col] = data[col] * scale

                elif col in ["val_loss", "loss", "val_non_zero_mse", "non_zero_mse",
                             "naive-model-non-zero", "naive-model-mse"]:
                    data[col] = (data[col] * scale * scale)
                    # continue
                # max_ = data[col].max()
                # data[col] = data[col] / max_

                plt.scatter(data['IC'].mean() - data['MC'], data["val_loss"],
                         alpha=(data["epoch"]/np.max(data["epoch"])).tolist(),
                            color=[colors_for_file[file]] * data["MC"].shape[0],
                            edgecolors=(1,1,1,0),
                            linewidths=0.1,
                            # label=slugify(str(files[file])),
                            # s=11
                            )

                plt.scatter(data['IC'].mean() - data['MC'][data.shape[0]-1], data["val_loss"][data.shape[0]-1],
                         alpha=(data["epoch"]/np.max(data["epoch"])).tolist()[data.shape[0]-1],
                            color=[colors_for_file[file]] * 1,
                            edgecolors=(1,1,1,0),
                            linewidths=0.1,
                            label=slugify(str(files[file])),
                            # s=11
                            )

                # plt.scatter(np.mean(data['MC']) , np.mean(data["val_loss"]),
                #             label=slugify(str(files[file])) + " mean",
                #             marker="*",
                #             color=colors_for_file[file],
                #             alpha=1,
                #             s=70)
                # plt.scatter(data['IC'].mean() - np.mean(data['MC'][10:]) , np.min(data["val_loss"]),
                #             label=slugify(str(files[file])) + " mean",
                #             marker="*",
                #             color=colors_for_file[file],
                #             alpha=1,
                #             s=70)
                # if idx == 0 :
                #     plt.scatter([data['IC'].mean()] * (1450-1150) , list (range(1150, 1450)),
                #                 label=" IC",
                #                 marker="o",
                #                 color=colors_for_file[file],
                #                 alpha=1,
                #                 s=1)



plt.legend(fontsize=10, loc="upper right")
# plt.yscale("log")
# plt.xscale("log")
# plt.gca().set_aspect(0.003)
# plt.ylim(1150, 1400)
# plt.ylim(1150, 1450)
# plt.xlim(0, 130)
plt.title("Val loss vs Model and Intrinsic Complexity")
plt.ylabel("Validation loss (MSE)")
plt.xlabel("|IC(Task)-MC(f|Task)| at various epochs")
plt.tight_layout()
plt.savefig("london_scatter_Val_vs_IC-MC-IO_LEN" + IO_len  + "-PRED_horiz_" + PRED_HORIZ + "Scale" + SCALE + ".png", dpi=300)
