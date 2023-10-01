import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os

# Directory containing the csv files
directory = "."

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

if os.path.exists("images"):
    os.system("rm -rf images")
os.mkdir("images")

plt.figure(figsize=(12, 8))

# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "CSR_MP_sum_y_exceeding_r_x_max": "-",
    "CSR_PM_sum_y_exceeding_r_x_max": "-",
    "val_loss": "--",
}

color_map = {file: plt.cm.jet(i/len(files)) for i, file in enumerate(files)}

handled_labels = []

# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)

    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1

    # Selecting columns of interest. Adjust if necessary
    columns = [
        "CSR_MP_sum_y_exceeding_r_x_max",
        "CSR_PM_sum_y_exceeding_r_x_max",
        "val_loss",
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
    plt.plot([], [], color=color, label=file)

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Combined Plot for All Files")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="upper left")
plt.tight_layout()

plt.savefig("images/combined_plot.png")  # Save the plot as an image
plt.show()
