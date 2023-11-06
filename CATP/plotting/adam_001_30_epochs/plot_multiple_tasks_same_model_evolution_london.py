import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os
import numpy as np

# Directory containing the csv files
directory = "."

# List of files
files = [
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-1-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-2-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-3-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-4-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-5-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-6-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-7-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-8-55-.csv",
]

# if os.path.exists("images"):
#     os.system("rm -rf images")
# os.mkdir("images")

# plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "MC": ":",
    "IC": "-",
    "val-MSE * 0.1": "--",
    "val-MSE": "--",
}

color_map = {file: plt.cm.jet(i/len(files)) for i, file in enumerate(files)}

handled_labels = []

# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["val-MSE * 0.1"] = df["val_loss"] / 10
    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1

    # Selecting columns of interest. Adjust if necessary
    columns = [
        "IC",
        "MC",
        "val-MSE * 0.1",
    ]
    for col in columns:

        line, = plt.plot(df['epoch'], df[col], linestyle=linestyles[col], color=color_map[file])

        # Add to handled labels for the legend
        if col not in handled_labels:

            handled_labels.append(col)

# Add dummy lines for legend
for col, ls in linestyles.items():
    plt.plot([], [], color='black', linestyle=ls, label=col)
for file, color in color_map.items():
    plt.plot([], [], color=color, label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
                                                     "").replace(".csv",""))

plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Value", fontsize=11)
plt.title(r"Combined Plot for London: $(i_0=4, p_h\in(1,8), s=55)$")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="best", fontsize=9)
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london.png", dpi=300)  # Save the plot as an image
plt.show()


plt.clf()
# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["val-MSE"] = df["val_loss"] / 10
    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1
    df["|IC-MC|"] = np.abs(df["IC"]) - np.abs(df["MC"])

    # Selecting columns of interest. Adjust if necessary
    columns = [
        # "IC",
        # "MC",
        "val-MSE",
    ]
    for col in columns:

        plt.scatter(df['|IC-MC|'], df[col], color=color_map[file], label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
                                                     "").replace(".csv","")[:-1])

        # Add to handled labels for the legend
        if col not in handled_labels:
            handled_labels.append(col)

plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Val MSE", fontsize=11)
plt.title(r"Combined Plot for Task: $(i_0=4, p_h\in(1,8), s=55, city=London)$")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="best", fontsize=9)
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london_scatter.png", dpi=300)  # Save the plot as an image
plt.show()
