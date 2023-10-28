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
    "MC_max": "yellow",
    "MC_min": "yellow",
}


for epoch in range(0, 30):
    # Dictionary to store y-values
    values = {
        "MC": [],
        "IC": [],
        "val_loss": [],
        "|IC-MC|": [],
        "MC_max": [],
        "MC_min": [],
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

        # Selecting columns of interest
        columns = ["IC", "MC", "val_loss", "|IC-MC|", "MC_max", "MC_min"]
        min_val = epoch # np.argmin(df["val_loss"])

        for col in columns:
            print (col, df.columns)
            if col == "val_loss":
                values[col].append(df[col][min_val] * 1)
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
    if epoch == 29:
        for col, y_values in values.items():
            if col == "val_loss":
                plt.scatter(x_values, y_values, label=slugify(col) , color=color_map[col]) # "*1"
            elif "MC_max" not in col and "MC_min" not in col:
                plt.scatter(x_values, y_values, label=col, color=color_map[col])

    for col, y_values in values.items():
        if col == "val_loss":
            plt.plot(x_values, y_values, color=color_map[col], alpha=(epoch+1) / 120)
        elif "MC_max" not in col and "MC_min" not in col:
            plt.plot(x_values, y_values, color=color_map[col], alpha=(epoch+1) / 120)
        # elif "MC_max" in col:
        #     plt.fill_between(x_values, values["MC_max"], values["MC_min"], color=color_map[col], alpha=0.4)

plt.xlabel("Prediction horizon", fontsize=10)
plt.title("Various metrics for taks of different prediction horizons", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
# plt.yscale("log")
plt.savefig("all_epochs_combined_plot_val_loss_vs_task_complexity__scatter.png")
plt.show()

