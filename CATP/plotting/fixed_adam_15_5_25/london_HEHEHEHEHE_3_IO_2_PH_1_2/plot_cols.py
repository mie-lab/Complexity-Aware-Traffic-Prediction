import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = [
    "combined.csv",
    "london-2-1-55-validation.csv",
    "london-2-2-55-validation.csv",
]

# pred_horiz = [1, 3, 5, 7, 1]
pred_horiz = ["Experiment", "Control-Easy", "Control-Hard"]


alphas = [1, 0.3, 0.6]

columns = [
            'epoch',
             # 'CSR_GB_sum',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             'CSR_MP_sum',
             # 'CSR_NM_sum',
             # 'CSR_PM_std',
             'CSR_PM_sum',
             # 'loss',
             # 'naive-model-mse',
             # 'naive-model-non-zero',
             # 'non_zero_mse',
             # 'self.CSR_GB_count',
             # 'self.CSR_NM_count',
             # 'self.CSR_PM_count',
             # 'val_loss',
             'val_non_zero_mse'
            ]
color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
color_dict["CSR_MP_sum"] = 'green'
color_dict["CSR_PM_sum"] = 'red'
color_dict['val_loss'] = 'blue'

# Loop through the CSV files
for idx, file in enumerate(files):

    if idx not in [
        0,
        1,
        2,
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

    if "combined" in file:
        lower_limit = data["val_non_zero_mse"]
        min_epoch_combined = data["epoch"][data["val_non_zero_mse"].idxmin()]
        min_epoch_combined_val = data["val_non_zero_mse"].min()

    if "london-2-1-55-validation" in file:
        upper_limit = data["val_non_zero_mse"]
        min_epoch_control = data["epoch"][data["val_non_zero_mse"].idxmin()]
        min_epoch_control_val = data["val_non_zero_mse"].min()

if min_epoch_combined is not None:
    plt.plot(min_epoch_combined, lower_limit.min(), 'r*', label="Experiment-" + str(round(min_epoch_combined_val,1)))

if min_epoch_control is not None:
    plt.plot(min_epoch_control, upper_limit.min(), 'ro', label="Control-" + str(round(min_epoch_control_val,1)))

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Sample data
x = range(20)
upper_limit = upper_limit[:20]
lower_limit = lower_limit[:20]

# Create a denser x-axis by introducing intermediate points
x_new = np.linspace(min(x), max(x), len(x)*30)  # Upsample by a factor of 10, for example

# Interpolate the y-values onto the new x-axis
f_upper = interp1d(x, upper_limit, kind='linear')  # You can also use 'cubic' or other methods
f_lower = interp1d(x, lower_limit, kind='linear')

upper_interp = f_upper(x_new)
lower_interp = f_lower(x_new)

# Plot and fill
plt.fill_between(x_new, upper_interp, lower_interp, where=(upper_interp > lower_interp), color="yellow", alpha=0.5)

# plt.yscale("log")
plt.ylim(1, 4000)

plt.title('Dataset switching experiment and control', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=6, ncol=2, loc="best")
plt.xticks(list(range(0, 30, 3)))

plt.grid(axis='x')

plt.savefig("val_non_zero_mse.png", dpi=300)
plt.show()
