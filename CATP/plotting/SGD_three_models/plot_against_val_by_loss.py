import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np
import glob
# Define the CSV filenames
# Define the CSV filenames
files = [
    "validation-create_model.csv",
    "validation-create_model_f_big.csv",
    "validation-create_model_small_epochs.csv"
 ]

pred_horiz = [ "deep_less_filter", "shallow_,more_filter", "deep_less_filter"] # ,"Control-Hard", "Control-Easy-High-LR", "Transfer-Hard-"]

alphas = np.array(range(1, len(pred_horiz) + 1)) / (len(pred_horiz) + 1)

columns = [
            'epoch',
             # 'CSR_GB_sum',
             # 'CSR_MP_count',
             # 'CSR_MP_std',
             'CSR_MP_sum',
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
        # 7,
        # 8,
        # 9
    ]:
        continue

    # Load the data
    data = pd.read_csv(file)

    data["val-by-loss"] = data["val_loss"] /  data["loss"]
    max_ = data["val-by-loss"].max()
    min_ = data["val-by-loss"].min()

    # data["val-by-loss"] = data["val-by-loss"] / max_
    data["val-by-loss"] = (data["val-by-loss"] - min_)/max_ * 2

    data = data.sort_values(by='val-by-loss', ascending=False)

    # Plot each column on the same plot
    n = 1

    linestyle = '-'


    for col in columns + ["val-by-loss"]:
        if idx == 0:
            data_col = data[col]
            if col not in ['val-by-loss', "epoch"]:

                rate_of_change = np.gradient(np.convolve(data_col, [1 / n] * n, "same"))
                print (rate_of_change)

                plt.plot(data['val-by-loss'], np.convolve(data_col, [1 / n] * n, "same"),
                         alpha=alphas[idx], color=color_dict[col],
                         label=col + "_rate_of_change_" + str(pred_horiz[idx]),
                         linestyle=linestyle)

        elif idx > 0:
            data_col = data[col]
            if col not in ['val-by-loss', "epoch"]:
                rate_of_change = np.gradient(np.convolve(data_col, [1 / n] * n, "same"))
                print(rate_of_change)

                plt.plot(data['val-by-loss'], np.convolve(data_col, [1 / n] * n, "same"),
                         alpha=alphas[idx], color=color_dict[col],
                         label=col + "_rate_of_change_" + str(pred_horiz[idx]),
                         linestyle=linestyle)


plt.title(r'X -axis: Ratio of $\frac{val}{train}$ loss', fontsize=8)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=8, ncol=2, loc="best")
# plt.xticks(list(range(0, 30, 1)), alpha=0.2)

plt.yscale("log")
# plt.xscale("log")

# plt.grid(axis='x')
plt.show()

# plt.xlim(n, (data['epoch'].max()) - n)
# plt.grid(axis='x')

plt.savefig("sorted_by_val_by_loss.png", dpi=300)
plt.show()
