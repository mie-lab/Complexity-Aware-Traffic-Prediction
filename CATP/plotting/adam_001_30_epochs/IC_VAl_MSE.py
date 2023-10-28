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
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-1-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-2-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-3-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-4-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-5-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-6-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-7-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modelmadrid-4-8-55-.csv"
]

# if os.path.exists("images"):
#     os.system("rm -rf images")
# os.mkdir("images")

# plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "CSR_MP_sum": "-",
    "CSR_PM_sum": "-",
    "val_loss": "--",
}

handled_labels = []

# plt.figure(figsize=(12, 8))

color_map = {
    "MC": "blue",
    "IC": "green",
    "val_loss": "orange",
    "|IC-MC|": "red",
    "KL-div": "yellow",
    "MC_max": "yellow",
    "MC_min": "yellow",
    r"$\frac{\sigma}{\mu}$":"yellow",
    "euclidean":"yellow",
}

# Dictionary to store y-values
values = {
    "MC": [],
    "IC": [],
    "val_loss": [],
    "|IC-MC|": [],
    "MC_max":[],
    "MC_min":[],
    # "KL-div":[],
    # r"$\frac{\sigma}{\mu}$":[],
    "euclidean":[]
}

# Iterate through each file and collect y-values
for file in files:
    filepath = os.path.join(directory, file)
    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["MC_min"] = df["MC"].min()
    df["MC_max"] = df["MC"].max()
    df["|IC-MC|"] = df["IC"] - df["MC"]

    df["MC"] = df["CSR_MP_sum"]
    mean_IC_mean = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    mean_IC_std = df["CSR_PM_std"][df["CSR_PM_sum"] > 0].mean()
    df["IC"] = mean_IC_mean
    df["IC_std"] = mean_IC_std

    mu2 = df["IC"]
    mu1 = df["MC"]
    sigma_2 = df["IC_std"]
    sigma_1 = df["CSR_MP_std"]

    # KL-div
    # df["KL-div"] = np.log(sigma_2 / sigma_1) + ((sigma_1 ** 2 + (mu2 - mu1) ** 2) / (2 * sigma_2 ** 2)) - 0.5

    # Bhattacharya
    # df["KL-div"] = 0.25 * (mu1-mu2) ** 2 / (sigma_1 ** 2 + sigma_2 **2 ) + 0.5 * np.log(
    #     (sigma_1 ** 2 + sigma_2 ** 2 )/(2 * sigma_1 * sigma_2)
    # )
    # df["KL-div"] = df["KL-div"] * 10000
    # df["euclidean"] = ((mu2-mu1) ** 2 + (sigma_2 - sigma_1) **2 ) ** 0.5
    df["euclidean"] = (np.abs(mu2-mu1) + np.abs(sigma_2 - sigma_1))
    # df["euclidean"] = ((mu2-mu1) ** 2 + (sigma_2 - sigma_1) **2 ) ** 0.5

    # print (df["KL-div"])
    df[r"$\frac{\sigma}{\mu}$"] = sigma_1/mu1 - sigma_2/mu2
    df[r"$\frac{\sigma}{\mu}$"] *= 1000
    print (df[r"$\frac{\sigma}{\mu}$"] * 1000)

    # Selecting columns of interest
    columns = ["IC", "MC", "val_loss", "MC_max", "MC_min", "|IC-MC|",
               # r"$\frac{\sigma}{\mu}$",
               "euclidean"] # "|IC-MC|", "KL-div",
    # min_val = np.argmin(df["val_loss"])
    val_loss = df["val_loss"]
    argmin = (val_loss).argsort()[:1]
    # min_val = np.argmin(df["KL-div"])

    for col in columns:
        print (col, df.columns)
        if col == "val_loss":
            # values[col].append(df[col][min_val] * 1)
            values[col].append(df[col][argmin].mean() * 1)
        else:
            # values[col].append(df[col][min_val])
            values[col].append(df[col][argmin].mean())

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
        plt.scatter(x_values, y_values, label=slugify(col) , color=color_map[col]) # "*1"
    elif "MC_max" not in col and "MC_min" not in col:
        plt.scatter(x_values, y_values, label=col, color=color_map[col])

for col, y_values in values.items():
    if col == "val_loss":
        plt.plot(x_values, y_values, color=color_map[col])
    elif "MC_max" not in col and "MC_min" not in col:
        plt.plot(x_values, y_values, color=color_map[col])
    elif "MC_max" in col:
        plt.fill_between(x_values, values["MC_max"], values["MC_min"], color=color_map[col], alpha=0)

plt.xlabel("Prediction horizon", fontsize=10)
plt.title("Various metrics for taks of different prediction horizons", fontsize=12)
plt.legend(fontsize=9)
plt.tight_layout()
# plt.yscale("log")
plt.savefig("combined_plot_val_loss_vs_task_complexity__scatter_CV.png")
plt.show()

