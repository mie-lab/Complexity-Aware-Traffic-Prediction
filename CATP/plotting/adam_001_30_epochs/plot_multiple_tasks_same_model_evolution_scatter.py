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

# Initialize empty lists to store data
ph = np.arange(1, 9)
IC = []
IC_MC = []
MC = []
Val_MSE = []

# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    # Find the epoch with the minimum val_loss
    min_val_loss_epoch = df.loc[df['val_loss'].idxmin()]
    argmin = np.argmin(df["val_loss"])

    # Compute the required metrics
    ic = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    mc = df["CSR_MP_sum"][argmin]
    val_mse = df["val_loss"][argmin]
    ic_mc = ic - mc

    # Append the metrics to the lists
    IC.append(ic)
    IC_MC.append(ic_mc)
    MC.append(mc)
    Val_MSE.append(val_mse)

# Create a scatter plot
# plt.figure(figsize=(12, 8))
SS = 62
plt.scatter(ph, IC, label='IC', color='blue',s=SS)
plt.scatter(ph, IC_MC, label='IC-MC', color='red',s=SS)
plt.scatter(ph, MC, label='MC', color='green',s=SS)
plt.scatter(ph, Val_MSE, label='Val-MSE', color='purple',s=SS)

# Create a line plot on top of the scatter plot
plt.plot(ph, IC, color='blue', alpha=0.5, linewidth=3)
plt.plot(ph, IC_MC, color='red', alpha=0.5, linewidth=3)
plt.plot(ph, MC, color='green', alpha=0.5, linewidth=3)
plt.plot(ph, Val_MSE, color='purple', alpha=0.5, linewidth=3)
variable_names = ["IC", "IC-MC", "MC", "Val-MSE"]

from scipy.stats import pearsonr

variables = [IC, IC_MC, MC, Val_MSE]
variable_names = ["IC", "IC-MC", "MC", "Val-MSE"]

for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        corr_coeff, _ = pearsonr(variables[i], variables[j])
        print(f'Correlation between {variable_names[i]} and {variable_names[j]}: {corr_coeff:.2f}')

# Label the axes
plt.xlabel('Prediction Horizon', fontsize=11)
plt.ylabel('Value', fontsize=11)

# Add a legend
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("London_scatter_IO4_ph_1-8_scale_55_UP_plot.png", dpi=300)
# Display the plot
plt.show()
