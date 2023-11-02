import matplotlib
import pandas as pd

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os
import numpy as np
# Directory containing the csv files
directory = "."
from slugify import slugify
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from slugify import slugify
from scipy.stats import spearmanr
from scipy.stats import pearsonr



# List of files
files = [
    "london-4-1-55-validation.csv",
    "london-4-2-55-validation.csv",
    "london-4-3-55-validation.csv",
    "london-4-4-55-validation.csv",
    "london-4-5-55-validation.csv",
    "london-4-6-55-validation.csv",
    "london-4-7-55-validation.csv",
    "london-4-8-55-validation.csv"
]

# if os.path.exists("images"):
#     os.system("rm -rf images")
# os.mkdir("images")

# plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "CSR_MP_sum_y_exceeding_r_x_max": "-",
    "CSR_PM_sum_y_exceeding_r_x_max": "-",
    "val_loss": "--",
}


handled_labels = []

# plt.figure(figsize=(12, 8))

color_map = {
    "MC": "blue",
    "IC": "green",
    "val_loss": "orange",
    "|IC-MC|": "red"
}

# Dictionary to store y-values
values = {
    "MC": [],
    "IC": [],
    "val_loss": [],
    "|IC-MC|": []
}

# Iterate through each file and collect y-values
for file in files:
    filepath = os.path.join(directory, file)
    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum_y_exceeding_r_x_max"]
    df["MC"] = df["CSR_MP_sum_y_exceeding_r_x_max"]
    df["|IC-MC|"] = df["IC"] - df["MC"]

    # Selecting columns of interest
    columns = ["IC", "MC", "val_loss", "|IC-MC|"]
    min_val = np.argmin(df["val_loss"])

    for col in columns:
        if col == "val_loss":
            values[col].append(df[col][min_val] * 10000)
        else:
            values[col].append(df[col][min_val])

# Compute and print Spearman correlation coefficients
columns_to_compare = list(values.keys())
for i in range(len(columns_to_compare)):
    for j in range(i + 1, len(columns_to_compare)):
        col1 = columns_to_compare[i]
        col2 = columns_to_compare[j]
        correlation, _ = spearmanr(values[col1], values[col2])
        print(f"Spearman correlation between {col1} and {col2}: {correlation:.3f}")


# Compute and print Spearman correlation coefficients
columns_to_compare = list(values.keys())
for i in range(len(columns_to_compare)):
    for j in range(i + 1, len(columns_to_compare)):
        col1 = columns_to_compare[i]
        col2 = columns_to_compare[j]
        correlation, _ = pearsonr(values[col1], values[col2])
        print(f"Pearson correlation between {col1} and {col2}: {correlation:.3f}")


# Plot the collected values
x_values = list(range(1, len(files) + 1))
for col, y_values in values.items():
    if col == "val_loss":
        plt.scatter(x_values, y_values, label=slugify(col) + "*10000", color=color_map[col])
    else:
        plt.scatter(x_values, y_values, label=col, color=color_map[col])

for col, y_values in values.items():
    if col == "val_loss":
        plt.plot(x_values, y_values, label=slugify(col) + "*10000", color=color_map[col])
    else:
        plt.plot(x_values, y_values, label=col, color=color_map[col])


plt.xlabel("Prediction horizon", fontsize=10)
plt.title("Various metrics for taks of different prediction horizons", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("images/combined_plot_val_loss_vs_task_complexity__scatter.png")
plt.show()

