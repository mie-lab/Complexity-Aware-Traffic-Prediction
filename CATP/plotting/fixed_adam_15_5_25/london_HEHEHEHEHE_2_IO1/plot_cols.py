import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np

# Define the CSV filenames
files = [
    "london-1-1-55-validation_control.csv",
    "combined.csv",
    "london-1-6-55-validation_control.csv",
]

# pred_horiz = [1, 3, 5, 7, 1]
pred_horiz = ["Control", "Experiment", "Control_hard"]


alphas = [0.5, 1, 0.6]

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
             'val_loss',
             # 'val_non_zero_mse'
            ]
color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
# color_dict["CSR_MP_sum_y_exceeding_r_x_max"] = 'green'
# color_dict["CSR_PM_sum_y_exceeding_r_x_max"] = 'red'
# color_dict['val_loss'] = 'blue'

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
                if "CSR" in col:
                    # Normalising required only for CX columns, not for validation columns
                    plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                             alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                             linestyle=linestyle)
                else:
                    plt.plot(data['epoch'], np.convolve(data[col] , [1/n]*n, "same"),  alpha=alphas[idx],
                             color=color_dict[col], label=col + "_" + str(pred_horiz[idx]),
                             linestyle=linestyle)

        if "combined.csv" in file:
            lower_limit = data["val_loss"]
        if "london-1-1-55-validation_control" in file:
            upper_limit = data["val_loss"]


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Sample data
x = range(20)
# upper_limit = [...]
# lower_limit = [...]

# Create a denser x-axis by introducing intermediate points
x_new = np.linspace(min(x), max(x), len(x)*10)  # Upsample by a factor of 10, for example

# Interpolate the y-values onto the new x-axis
f_upper = interp1d(x, upper_limit, kind='linear')  # You can also use 'cubic' or other methods
f_lower = interp1d(x, lower_limit, kind='linear')

upper_interp = f_upper(x_new)
lower_interp = f_lower(x_new)

# Plot and fill
plt.fill_between(x_new, upper_interp, lower_interp, where=(upper_interp > lower_interp), color="yellow", alpha=0.5)

plt.show()


plt.show()
plt.title('Dataset switching experiment and control', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=8, ncol=2, loc="upper right")
plt.xticks(list(range(0, 30, 3)))

# Display the plot
# plt.yscale("symlog")
# plt.ylim(0, 2000)
plt.savefig("val_loss_2.png", dpi=300)
plt.show()
