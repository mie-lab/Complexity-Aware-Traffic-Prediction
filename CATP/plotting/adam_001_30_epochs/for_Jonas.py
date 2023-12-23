import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from smartprint import smartprint as sprint

# Directory containing the csv files
directory = "."

# List of files
files = [
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-1-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-2-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-3-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-4-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-5-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-6-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-7-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-8-55-.csv",
]

# if os.path.exists("images"):
#     os.system("rm -rf images")
# os.mkdir("images")

# plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "Model Complexity": "-",
    "Intrinsic Complexity": "--",
    "Validation Error": ":",
    # "val-MSE": "--",
}

color_map = {

    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-1-55-.csv" : "tab:orange",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-2-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-3-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-4-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-5-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-6-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-7-55-.csv" : "tab:blue"
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-8-55-.csv",

}

handled_labels = []


hard_ness_label = {
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-1-55-.csv" : "15-minute prediction",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-7-55-.csv" : "2 hour prediction"
}

plt.plot(range(30), [0]*30, alpha=0) # dummy line to ensure placeholder
plt.xlim(0, 30)
plt.ylim(0, 5000)
plt.plot(range(30), [3000]*30, color="tab:blue", linewidth=1.5,  label="Hard task", linestyle="--")
plt.plot(range(30), [1500]*30, color="tab:orange", linewidth=1.5,  label="Easy task", linestyle="--")
plt.plot([], [], color="black", linewidth=1.5, label="Intrinsic Complexity", linestyle="--")
# Iterate through each file
for EPOCH_COUNTER in range(1, 30):
    for counter, file in enumerate(files):
        filepath = os.path.join(directory, file)

        # Read the CSV
        df = pd.read_csv(filepath)
        df["Intrinsic Complexity"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
        df["Model Complexity"] = df["CSR_MP_sum"] * 4
        df["Validation Error"] = df["val_loss"]

        # Adjust the 'epoch' column
        df['epoch'] = df['epoch'] + 1

        # Selecting columns of interest. Adjust if necessary
        columns = [
            "Intrinsic Complexity",
            "Model Complexity",
            # "Validation Error",
        ]
        for colcounter, col in enumerate(columns):
            if col != "Intrinsic Complexity":
                if counter == 0 and EPOCH_COUNTER == 1:
                    plt.plot(df['epoch'][:EPOCH_COUNTER], df[col][:EPOCH_COUNTER], linestyle=linestyles[col], color=color_map[file], label=col, linewidth=2)
                    # plt.scatter(df['epoch'][:EPOCH_COUNTER], df[col][:EPOCH_COUNTER], color=color_map[file], marker="s", s=50)
                    if colcounter == 0:
                        plt.plot([], [], label="15 minute prediction", color="tab:orange", linewidth=1.5)
                        plt.plot([], [], label="1 hour prediction", color="tab:blue", linewidth=1.5)
                else:
                    plt.plot(df['epoch'][:EPOCH_COUNTER], df[col][:EPOCH_COUNTER], linestyle=linestyles[col], color=color_map[file])


        plt.xlabel("Epochs", fontsize=11)
        plt.ylabel("Value", fontsize=11)
        plt.title(r"Evolution of Model Complexity for two tasks")
        # plt.yscale("log")  # Logarithmic scale for y-axis
        plt.legend(ncol=2, loc="best", fontsize=11)
        plt.tight_layout()

        plt.savefig("DEMO_plot_for_JONAS"+str(EPOCH_COUNTER).zfill(2)+".png", dpi=300)  # Save the plot as an image
# plt.show()
