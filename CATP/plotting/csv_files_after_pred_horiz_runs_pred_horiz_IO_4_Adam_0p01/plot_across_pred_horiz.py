import pandas as pd
import matplotlib
from scipy.stats import spearmanr

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os

# Directory containing the csv files
directory = "."

# List of files
files = [
    # "london-1-1-55-validation.csv",
    "london-1-2-55-validation.csv",
]

if os.path.exists("images"):
    os.system("rm -rf images")
os.mkdir("images")

plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "CSR_MP_sum_y_exceeding_r_x_max": "-",
    "CSR_PM_sum_y_exceeding_r_x_max": "-",
    "loss": "--",
    "val_loss": "-."
}

color_map = {file: plt.cm.jet(i/len(files)) for i, file in enumerate(files)}

def get_label(col, file):
    if col == "CSR_MP_sum_y_exceeding_r_x_max":
        return f"MC({os.path.splitext(file)[0]})"
    elif col == "CSR_PM_sum_y_exceeding_r_x_max":
        return f"IC({os.path.splitext(file)[0]})"
    else:
        return col

handled_labels = []

# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)

    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1

    df["CSR_MP_std"] = df["CSR_MP_std"] * 0
    df["CSR_PM_std"] = df["CSR_PM_std"] * 0

    df['MP_std_low']  = df["CSR_MP_std"] + df["CSR_MP_sum_y_exceeding_r_x_max"]
    df['PM_std_low'] = df["CSR_MP_std"] + df["CSR_PM_sum_y_exceeding_r_x_max"]
    df['MP_std_high']  = - df["CSR_MP_std"] + df["CSR_MP_sum_y_exceeding_r_x_max"]
    df['PM_std_high'] = - df["CSR_MP_std"] + df["CSR_PM_sum_y_exceeding_r_x_max"]

    # Compute Spearman correlation
    correlation, p_value = spearmanr(df["CSR_MP_sum_y_exceeding_r_x_max"], df["loss"])
    print(f"Spearman correlation for {file} between MC and Val loss: {correlation:.2f}")

    # Selecting columns of interest
    columns = [
        "CSR_MP_sum_y_exceeding_r_x_max",
        "CSR_PM_sum_y_exceeding_r_x_max",
        "loss",
        "val_loss"
    ]
    for col in columns:
        line, = plt.plot(df['epoch'], df[col], linestyle=linestyles[col], color=color_map[file],
                         label=get_label(col, file))

        # Fill between for MP and PM
        if col == "CSR_MP_sum_y_exceeding_r_x_max":
            plt.fill_between(df['epoch'], df['MP_std_low'], df['MP_std_high'], color=color_map[file],
                             alpha=0.3)  # Use alpha for transparency
        elif col == "CSR_PM_sum_y_exceeding_r_x_max":
            plt.fill_between(df['epoch'], df['PM_std_low'], df['PM_std_high'], color=color_map[file],
                             alpha=0.3)  # Use alpha for transparency

        # Add to handled labels for the legend
        if col not in handled_labels:
            handled_labels.append(col)


plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Combined Plot for All Files")
plt.grid()
# plt.yscale("symlog")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="upper left")
plt.ylim(0, 2500)
plt.tight_layout()

plt.savefig("images/combined_plot.png")  # Save the plot as an image
plt.show()
