import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np
from slugify import slugify


# Define the CSV filenames
files = {
    "validation-create_model_f_def_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "55" + "-.csv": "f_mid_easy_task",
    "validation-create_model_f_def_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "85" + "-.csv": "f_mid_hard_task",
    "validation-create_model_f_shallow_1_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "55" + "-.csv": "f_shallow_1_easy_task",
    "validation-create_model_f_shallow_1_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "85" + "-.csv": "f_shallow_1_hard_task",
    "validation-create_model_f_shallow_2_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "55" + "-.csv": "f_shallow_2_easy_task",
    "validation-create_model_f_shallow_2_no_BN-adam-0001-london-" + "4" + "-" + "6" + "-" + "85" + "-.csv": "f_shallow_2_hard_task",
}


colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]  # these are 10 distinguishable colors

# Create a dictionary mapping each file to a color
colors_for_file = {files[file].replace("easy_task","").replace("hard_task","")
                   : colors[i % len(colors)] for i, file in enumerate(files)}

alphas = [0.1, 0.2,  0.4, 0.8, 0.9, 1] # , 0.1, 0.5]
# alphas = np.array([1,2,3,4,5,6])/6
# alphas = [0.3, 1]
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
             # 'naive-model-mse',
             # 'naive-model-non-zero',
             # 'non_zero_mse',
             # 'self.CSR_GB_count',
             # 'self.CSR_NM_count',
             # 'self.CSR_PM_count',
             # 'val_loss',
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
                         color=colors_for_file[files[file].replace("easy_task","").replace("hard_task","")],
                         label=slugify( col + "_" + str(files[file])).replace("mc-", "MC-"),
                         linestyle=linestyle)



plt.title('MC evaluation during training', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
# plt.legend(fontsize=6, ncol=2, loc="upper right")
plt.legend(fontsize=8, ncol=2, loc="best")
plt.xticks(list(range(0, 30, 1)), rotation=90, fontsize=4)
# plt.yscale("log")

plt.grid(axis='x',alpha=0.05)
# plt.xlim(3//2, 30-n)
# plt.xlim(0, 10)
# plt.ylim(1, 3000)
plt.tight_layout()

plt.savefig("london-IO_LEN" + "_two_tasks" + ".png", dpi=300)
plt.show()
