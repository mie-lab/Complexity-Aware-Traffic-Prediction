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
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-2-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-3-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-4-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-5-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-6-55-.csv",
    "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-7-55-.csv",
    # "validation-default_model--adam-0p1-different-tasks-one-modellondon-4-8-55-.csv",
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
    # "val-MSE": "--",
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
plt.title(r"Evolution of MC Task: $(i_0=4, p_h\in(1,8), s=55, city=London)$")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="best", fontsize=9)
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london.png", dpi=300)  # Save the plot as an image
plt.show()

plt.clf()
# Set line styles to differentiate between `val_loss` and other columns
linestyles = {
    "MC@val-MSE min": "-",
    "IC": "-",
    "val-MSE min": "-",
    # "val-MSE": "--",
}
marker_dict = {
    "MC@val-MSE min": "s",
    "IC": "^",
    "val-MSE min": "o",
    # "val-MSE": "--",
}

color_map = {file: plt.cm.jet(i / len(files)) for i, file in enumerate(files)}

handled_labels = []

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["val-MSE"] = df["val_loss"] /50

    df["val-MSE min"] = df["val-MSE"].cummin()
    indices_where_activated = df.index[df["val-MSE min"] == df["val-MSE"]].tolist()



    filtered_quantity = df.loc[indices_where_activated, "MC"]
    df["MC@val-MSE min"] = filtered_quantity

    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1

    # Selecting columns of interest. Adjust if necessary
    columns = [
        "IC",
        "val-MSE min",
        "MC@val-MSE min",
    ]
    for col in columns:
        if col == "MC@val-MSE min":
            line, = plt.plot(df['epoch'][indices_where_activated], df[col][indices_where_activated],
                             linestyle=linestyles[col], color=color_map[file])
            plt.scatter(df['epoch'][indices_where_activated], df[col][indices_where_activated], marker=marker_dict[col],
                        color=color_map[file])
        elif col == "IC":
            line, = plt.plot(df['epoch'][:], df[col][:],
                             linestyle=linestyles[col], color=color_map[file])
            # ax1.scatter(df['epoch'][:], df[col][:], marker=marker_dict[col],
            #             color=color_map[file])
        if col == "val-MSE min":
            plt.scatter(df['epoch'][indices_where_activated], df[col][indices_where_activated], marker=marker_dict[col],
                        color=color_map[file])
            plt.plot(df['epoch'][indices_where_activated], df[col][indices_where_activated], marker=marker_dict[col],
                    color=color_map[file])
        # Add to handled labels for the legend
        if col not in handled_labels:
            handled_labels.append(col)

# Add dummy lines for legend
for col, ls in linestyles.items():
    plt.plot([], [], color='black', linestyle=ls, label=col)
for file, color in color_map.items():
    plt.plot([], [], color=color, label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
                                                     "").replace(".csv", ""))

plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Value", fontsize=11)
plt.title(r"Evolution of MC Task: $(i_0=4, p_h\in(1,8), s=55, city=London)$")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend(ncol=2, loc="best", fontsize=9)
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london_cummin.png", dpi=300)  # Save the plot as an image
plt.show()

plt.clf()
# Iterate through each file
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["val-MSE"] = df["val_loss"] #   / 10
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

plt.xlabel("|IC-MC|", fontsize=11)
plt.ylabel("Val MSE", fontsize=11)
plt.title(r"Task: $(i_0=4, p_h\in(1,8), s=55, city=London)$")
# plt.yscale("log")  # Logarithmic scale for y-axis
plt.ylim(500, 2700)
plt.legend(ncol=2, loc="best", fontsize=10)
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london_scatter.png", dpi=300)  # Save the plot as an image
plt.show()



plt.clf()
# Iterate through each file
X_MC = []
Y_MC = []
X_IC = []
Y_IC = []
X_MC_IC = []
Y_MC_IC = []


for ph_counter, file in enumerate(files):
    # Removed MC plot, only retained labels
    plt.scatter([], [], color=color_map[file], label=r'$p_h$=' + str(ph_counter + 1), marker="s")

MC_values_record = []
IC_Values_record = []
for file in files:
    filepath = os.path.join(directory, file)

    # Read the CSV
    df = pd.read_csv(filepath)
    df["IC"] = df["CSR_PM_sum"][df["CSR_PM_sum"] > 0].mean()
    df["MC"] = df["CSR_MP_sum"]
    df["val-MSE"] = df["val_loss"] #   / 10
    # Adjust the 'epoch' column
    df['epoch'] = df['epoch'] + 1
    df["|IC-MC|"] = np.abs(df["IC"]) - np.abs(df["MC"])

    # Selecting columns of interest. Adjust if necessary
    # columns = [
    #     # "IC",
    #     # "MC",
    #     "val-MSE",
    # ]
    argmin = np.argsort(df["val_loss"])[0]



    for col in ["val-MSE"]:
        MC_values_record.append(df["MC"][argmin])
        IC_Values_record.append(df["IC"][argmin])
        plt.scatter(df['MC'][argmin], df[col][argmin], color=color_map[file],
                    # label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
                    #                                  "").replace(".csv","")[:-1].title() + " MC",
                    marker="o",
                    s=70
                    )

        # Removed MC plot, only retained labels

        # if argmin has more than one items
        # X_MC.extend(df['MC'][argmin].tolist())
        # Y_MC.extend(df[col][argmin].tolist())

        # to be used if argmin has one item
        sprint(df['MC'][argmin])
        X_MC.extend([df['MC'][argmin]])
        Y_MC.extend([df[col][argmin]])


        if file == files[0]:
            plt.scatter(df['IC'][argmin], df[col][argmin], color=color_map[file],
            #             label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
            #                                              "").replace(".csv","")[:-1].title() + " IC",
                                                                     marker='*', s=70)

            # diagrregated legend
            plt.scatter([], [],
                        label=r"$IC$", marker='*', color="black", s=70)
            plt.scatter([], [],
                        label=r"$MC$", marker='o', color="black", s=70)
        else:
            plt.scatter(df['IC'][argmin], df[col][argmin], color=color_map[file], marker='*', s=70)

        # if argmin is an iterable
        # X_IC.extend(df['IC'][argmin].tolist())
        # Y_IC.extend(df[col][argmin].tolist())

        # if not iterable
        X_IC.extend([df['IC'][argmin]])
        Y_IC.extend([df[col][argmin]])

        if file == files[0]:
            plt.scatter(df['|IC-MC|'][argmin], df[col][argmin], color=color_map[file],
                        # label=file.replace("validation-default_model--adam-0p1-different-tasks-one-model",
                        #                                  "").replace(".csv","")[:-1].title() + " |IC-MC|",
                        marker='^', s=70)

            # diagrregated legend
            plt.scatter([], [],
                        label=r"$|IC-MC|$", marker='^', color="black", s=70)
        else:
            plt.scatter(df['|IC-MC|'][argmin], df[col][argmin], color=color_map[file], marker='^', s=70)


        # if argmin is an iterable
        # X_MC_IC.extend(df['|IC-MC|'][argmin].tolist())
        # Y_MC_IC.extend(df[col][argmin].tolist())

        # if argmin is single value (scalar)
        X_MC_IC.extend([df['|IC-MC|'][argmin]])
        Y_MC_IC.extend([df[col][argmin]])

        # Add to handled labels for the legend
        if col not in handled_labels:
            handled_labels.append(col)


correlation_coef, p_value = pearsonr(np.array(X_MC), np.array(Y_MC))
# plt.plot(X_MC, LinearRegression().fit(np.array(X_MC).reshape(-1,1), np.array(Y_MC).reshape(-1,1))
#          .predict(np.array(X_MC).reshape(-1,1)), '-', color="black",alpha=0.6)
print(f"MC Pearson Correlation Coefficient: {correlation_coef}")
print(f"P-value: {p_value}")

correlation_coef, p_value = pearsonr(np.array(X_IC), np.array(Y_IC))
plt.plot(X_IC, LinearRegression().fit(np.array(X_IC).reshape(-1,1), np.array(Y_IC).reshape(-1,1))
         .predict(np.array(X_IC).reshape(-1,1)), '-', color="black", alpha=0.4, linewidth=2)
print(f"IC Pearson Correlation Coefficient: {correlation_coef}")
print(f"P-value: {p_value}")

correlation_coef, p_value = pearsonr(np.array(X_MC_IC), np.array(Y_MC_IC))
plt.plot(X_MC_IC, LinearRegression().fit(np.array(X_MC_IC).reshape(-1,1), np.array(Y_MC_IC).reshape(-1,1))
         .predict(np.array(X_MC_IC).reshape(-1,1)), '-', color="black", alpha=0.4, linewidth=2)
print(f"IC-MC Pearson Correlation Coefficient: {correlation_coef}")
print(f"P-value: {p_value}")

max_MC_values_record, min_MC_values_record = max(MC_values_record), min(MC_values_record)
x1, x2 = min_MC_values_record, max_MC_values_record
# plt.axvspan(0, x1, color='green', alpha=0.02)  # less complex
plt.axvspan(x1, x2, color='blue', alpha=0.05)  # just enough complex
# plt.axvspan(x2, 270, color='red', alpha=0.07)  # too complex

# plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.7, r"$\Delta MC=$" + str(round(max_MC_values_record-min_MC_values_record)),
#          ha='center')

# Add a horizontal line with end markers
y_value = plt.ylim()[1] * 0.7 - 15  # Set the y-value where you want the horizontal line
plt.hlines(y=y_value, xmin=x1, xmax=x2, colors='black', linestyles='--', lw=2)
# plt.plot([x1, x2], [y_value, y_value], 'o')  # End markers

# Add small vertical lines as end markers
marker_height = 20  # Define the height of your markers
plt.vlines(x1, y_value - marker_height / 2, y_value + marker_height / 2, colors='black', linestyles='solid', lw=2)
plt.vlines(x2, y_value - marker_height / 2, y_value + marker_height / 2, colors='black', linestyles='solid', lw=2)

# Add text
delta_MC = round(max_MC_values_record - min_MC_values_record)
plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.7, r"$\Delta~(MC)=$" + str(delta_MC), ha='center')
plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.7 - 70, r"$\mu~(MC)=$" + str(round(np.mean(MC_values_record))), ha='center')



max_MC_values_record, min_MC_values_record = max(IC_Values_record), min(IC_Values_record)
x1, x2 = min_MC_values_record, max_MC_values_record
# plt.axvspan(0, x1, color='green', alpha=0.02)  # less complex
plt.axvspan(x1, x2, color='blue', alpha=0.05)  # just enough complex
# plt.axvspan(x2, 270, color='red', alpha=0.07)  # too complex

# plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.7, r"$\Delta MC=$" + str(round(max_MC_values_record-min_MC_values_record)),
#          ha='center')

# Add a horizontal line with end markers
y_value = plt.ylim()[1] * 0.3 + 120  # Set the y-value where you want the horizontal line
plt.hlines(y=y_value, xmin=x1, xmax=x2, colors='black', linestyles='--', lw=2)
# plt.plot([x1, x2], [y_value, y_value], 'o')  # End markers

# Add small vertical lines as end markers
marker_height = 20  # Define the height of your markers
plt.vlines(x1, y_value - marker_height / 2, y_value + marker_height / 2, colors='black', linestyles='solid', lw=2)
plt.vlines(x2, y_value - marker_height / 2, y_value + marker_height / 2, colors='black', linestyles='solid', lw=2)

# Add text
delta_MC = round(max_MC_values_record - min_MC_values_record)
plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.4, r"$\Delta~(IC)=$" + str(delta_MC), ha='center')
plt.text((plt.xlim()[0] + x1 + x2) / 2, plt.ylim()[1] * 0.4 - 70, r"$\mu~(IC)=$" + str(round(np.mean(IC_Values_record))), ha='center')




plt.xlabel("  $MC$, $|IC-MC|$, and $IC$", fontsize=11)
plt.ylabel("Val MSE", fontsize=11)
plt.title(r"Task: $(i_0=4, p_h\in(1,7), s=55, city=London)$", fontsize=13)
# plt.yscale("log")  # Logarithmic scale for y-axis
# plt.ylim(500, 2400)

# plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.25, 1.4), fontsize=11)

plt.gca().set_aspect(1.6)
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.51, 1.3), fontsize=10.6)
 # loc="upper center",
plt.tight_layout()

plt.savefig("evolution_multiple_tasks_london_scatter_one_epoch_min_val_loss_min_Val.png", dpi=300)  # Save the plot as an image
plt.show()
