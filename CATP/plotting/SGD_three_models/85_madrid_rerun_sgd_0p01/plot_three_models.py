import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = {
    "validation-create_model_f_big_reg_NO_bn-sgd-0001-madrid-1-4-85-.csv": "f_BIG_hard",
    # "validation-create_model_f_big_reg_NO_bn-sgd-0001-madrid-2-4-55-.csv":"f_BIG_easy",
    "validation-create_model_f_def_no_BN-sgd-0001-madrid-1-4-85-.csv": "f_small_hard",
    # "validation-create_model_f_def_no_BN-sgd-0001-madrid-2-4-55-.csv": "f_small_easy"
}




alphas = [1, 0.5, 0.25] # , 0.1, 0.5]

columns = [
            'epoch',
             # 'CSR_GB_sum',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             # 'CSR_MP_sum',
             # 'CSR_NM_sum',
             # 'CSR_PM_std',
             # 'CSR_PM_sum',
             'loss',
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

            if col != 'epoch':

                scale = 5000

                if col in ["MC", "IC"]:
                    data[col] = data[col] * scale
                # if col in ["IC"]:
                #     data[col] = data[col] * 1 # Since we don't use any Data loader in IC computation
                elif col in ["val_loss", "loss", "val_non_zero_mse", "non_zero_mse",
                             "naive-model-non-zero", "naive-model-mse"]:
                    data[col] = data[col] * scale * scale
                    # continue
                # max_ = data[col].max()
                # data[col] = data[col] / max_

                plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                         alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(files[file]),
                         linestyle=linestyle)



plt.title('MC evaluation during training', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
# plt.legend(fontsize=6, ncol=2, loc="upper right")
plt.legend(fontsize=6, ncol=2, loc="best")
plt.xticks(list(range(0, 100, 1)), rotation=90, fontsize=4)
# plt.yscale("log")
plt.grid(axis='x',alpha=0.05)
# plt.xlim(n, 50-n)
# plt.ylim(1, 4000)
plt.tight_layout()

plt.savefig("scale_one_task_two_models.png", dpi=300)
plt.show()
