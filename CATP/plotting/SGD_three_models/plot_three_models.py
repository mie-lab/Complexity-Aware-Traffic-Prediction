import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = [
    "validation-create_model.csv",
    "validation-create_model_f_big.csv",
    "validation-create_model_small_epochs.csv"
 ]

pred_horiz = [ "deep_less_filter", "shallow_,more_filter", "deep_less_filter"] # ,"Control-Hard", "Control-Easy-High-LR", "Transfer-Hard-"]


alphas = [1, 1, 1] # , 0.1, 0.5]

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
             # 'naive-model-mse',
             # 'naive-model-non-zero',
             # 'non_zero_mse',
             # 'self.CSR_GB_count',
             # 'self.CSR_NM_count',
             # 'self.CSR_PM_count',
             'val_loss',
             # 'val_non_zero_mse'
            ]

columns = columns + ["MC", "IC"] #["val/train"]



color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
color_dict["IC"] = 'green'
color_dict["MC"] = 'red'
color_dict['val_loss'] = 'blue'

# Loop through the CSV files
for idx, file in enumerate(files):

    if idx not in [
        0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
    ]:
        continue

    # Load the data
    data = pd.read_csv(file)

    # Plot each column on the same plot
    n = 1


    # data["val/train"] = data["val_non_zero_mse"] / data["non_zero_mse"]
    data["MC"] = data["CSR_MP_sum"]
    data["IC"] = data["CSR_PM_sum"]

    linestyle = '-'
    # if "create_model_f_big" in file:
    #     linestyle = ":"
    for col in columns:

        if idx >= 0 :
            # Exclude 'epoch' column from plotting

            if col != 'epoch':

                if col in ["MC"]:
                    data[col] = data[col] * 10 ** 4
                if col in ["IC"]:
                    data[col] = data[col] * 10 ** 0 # Since we don't use any Data loader in IC computation
                elif col in ["val_loss", "loss"]:
                    data[col] = data[col] * 10 ** 8
                    # continue
                # max_ = data[col].max()
                # data[col] = data[col] / max_

                plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                         alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                         linestyle=linestyle)



plt.title('MC evaluation during training', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=6, ncol=2, loc="upper right")
plt.xticks(list(range(0, 30, 1)), rotation=90)
plt.yscale("log")
plt.grid(axis='x')

plt.savefig("three_models_evolution_1.png", dpi=300)
plt.show()
