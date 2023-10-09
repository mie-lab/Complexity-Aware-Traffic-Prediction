import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = [
    # "combined.csv",
    # "london-4-1-55-validation_control.csv",
    # "london-4-1-55-validation-high-LR_0p01.csv",
    # "london-4-1-55-validation_high_LR_0p001_0999_09999.csv",
    # "london-4-1-55-validation_pred_horiz_hard_to_easy_example_control.csv",
    # "london-4-1-55-validation_sampling_case.csv"
    "london-4-2-55-validation.csv",
]

# pred_horiz = [1, 3, 5, 7, 1]
pred_horiz = [ "Task-PH_1", "Task_PH_2"] # ,"Control-Hard", "Control-Easy-High-LR", "Transfer-Hard-"]


alphas = [1, 0.45] # , 0.1, 0.5]

columns = [
            'epoch',
             # 'CSR_GB_sum',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             'CSR_MP_sum',
             # 'CSR_NM_sum',
             # 'CSR_PM_std',
             'CSR_PM_sum',
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
color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
color_dict["CSR_MP_sum"] = 'green'
color_dict["CSR_PM_sum"] = 'red'
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

    linestyle = '-'
    if "validation_experiment.csv" in file:
        linestyle = ":"
    for col in columns:

        if idx >= 0 :
            # Exclude 'epoch' column from plotting

            if col != 'epoch':
                plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                         alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                         linestyle=linestyle)



plt.title('Dataset switching experiment and control', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=6, ncol=2, loc="best")
plt.xticks(list(range(0, 20, 1)))
# plt.yscale("log")
plt.grid(axis='x')

plt.savefig("lr_test_train_loss.png", dpi=300)
plt.show()
