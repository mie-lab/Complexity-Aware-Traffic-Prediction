import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = [
    "london-4-1-55-validation_control.csv",
    "london-4-6-55-validation_control.csv",
    "london-4-7-55-validation_control.csv",
    "london-4-1-55-validation_experiment.csv",
]

# pred_horiz = [1, 3, 5, 7, 1]
pred_horiz = [1, 6, 7, 1]

alphas = np.array(pred_horiz)/7  # Different alpha values for each CSV

columns = [
            'epoch',
             # 'CSR_GB_sum_y_exceeding_r_x_max',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             'CSR_MP_sum_y_exceeding_r_x_max',
             # 'CSR_NM_sum_y_exceeding_r_x_max',
             # 'CSR_PM_std',
             'CSR_PM_sum_y_exceeding_r_x_max',
             # 'loss',
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
color_dict["CSR_MP_sum_y_exceeding_r_x_max"] = 'green'
color_dict["CSR_PM_sum_y_exceeding_r_x_max"] = 'red'
color_dict['val_loss'] = 'blue'

# Loop through the CSV files
for idx, file in enumerate(files):

    if idx not in [
        0,
        1,
        2,
        3,
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
        if idx == 0:
            # Exclude 'epoch' column from plotting

            if col != 'epoch':
                if "CSR" in col:
                    # Normalising required only for CX columns, not for validation columns
                    plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                             alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                             linestyle=linestyle)
                else:
                    plt.plot(data['epoch'], np.convolve(data[col] , [1/n]*n, "same"),  alpha=alphas[idx],
                             color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                             linestyle=linestyle)

        else:
            if col != 'epoch':
                if "CSR" in col:
                    # Normalising required only for CX columns, not for validation columns
                    plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                             alpha=alphas[idx], color=color_dict[col], linestyle=linestyle)
                else:
                    plt.plot(data['epoch'], np.convolve(data[col] , [1/n]*n, "same"),  alpha=alphas[idx],
                             color=color_dict[col], linestyle=linestyle)

# Add title, xlabel, ylabel and legend
plt.title('Metrics Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=8, ncol=2)
plt.xticks(list(range(0, 50, 3)))

# Display the plot
# plt.yscale("log")
plt.savefig("all_columns_image.png")
plt.show()