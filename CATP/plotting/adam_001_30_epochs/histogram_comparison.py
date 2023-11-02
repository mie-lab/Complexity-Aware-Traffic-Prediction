import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
from scipy.stats import pearsonr

for p_h in [1, 2, 3, 4, 6, 7, 8, 9]:
    P_H = str(p_h)
    # Load the CSV data
    data = pd.read_csv('record_distribution_ph' + P_H + '.csv', header=None)

    filename = "validation-default_model--adam-0p1-recorededdifferent-tasks-one-modellondon-4-"
    if p_h >= 5:
        filename = filename.replace("recoreded", "recoreded_reverse")

    val_loss = pd.read_csv(filename + P_H +
                           "-55-.csv")["val_loss"]
    # Separate the data based on MP and PM markers
    mp_data = data[data[0] == 'MP'].iloc[:, 1:].values
    pm_data = data[data[0] == 'PM'].iloc[:, 1:].values

    N = 50

    # Concatenate all PM rows and then compute a single histogram
    pm_aggregated = pm_data.ravel()
    pm_histogram = np.histogram(pm_aggregated, bins=N)[0]
    pm_bins = np.histogram(pm_aggregated, bins=N)[1]

    # Compute histograms for MP (assuming 10 bins for simplicity)
    mp_histograms = [np.histogram(row, bins=pm_bins)[0] for row in mp_data]



    # plt.plot(pm_histogram[1:], label="PM")
    # # sprint (np.mean(pm_data.ravel()))
    # for counter, mp_hist in enumerate(mp_histograms):
    #     # sprint(np.mean(mp_data[counter, 1:].ravel()))
    #     plt.plot(mp_hist[1:], label="MP"+ str(counter + 1), alpha=(counter+1)/28, color="red")
    # plt.legend()
    # plt.xlim(1, N)
    # sprint (np.histogram(pm_aggregated, bins=N)[1])
    # plt.xticks(np.histogram(pm_aggregated, bins=N)[1])
    # sprint (np.histogram(pm_aggregated, bins=N)[0])
    # plt.yscale("symlog")
    # plt.savefig("actual_hists.png")

    # Compute the distance between each MP histogram and the PM histogram
    # Here, we use the Euclidean distance
    # distances = [np.linalg.norm(mp_hist[1:] - pm_histogram[1:]) for mp_hist in mp_histograms]
    distances = [wasserstein_distance(mp_hist[1:], pm_histogram[1:]) for mp_hist in mp_histograms]
    # sprint (len(distances), val_loss.shape)
    plt.scatter(distances, val_loss, label=r"$p_h=$" + P_H)
    sprint (pearsonr(distances, val_loss), spearmanr(distances, val_loss))
    # plt.clf()
    # plt.plot(distances)
plt.xlabel("Wasserstein |IC-MC|")
plt.ylabel("Val loss")
plt.legend()
plt.tight_layout()
plt.savefig("histcompare_different_PH_Val_loss.png")


plt.clf()
for p_h in [1, 2, 3, 4, 6, 7, 8, 9]:
    P_H = str(p_h)
    # Load the CSV data
    data = pd.read_csv('record_distribution_ph' + P_H + '.csv', header=None)

    filename = "validation-default_model--adam-0p1-recorededdifferent-tasks-one-modellondon-4-"
    if p_h >= 5:
        filename = filename.replace("recoreded", "recoreded_reverse")

    val_loss = pd.read_csv(filename + P_H +
                           "-55-.csv")["val_loss"]
    argmin = np.argmin(val_loss)

    # Separate the data based on MP and PM markers
    mp_data = data[data[0] == 'MP'].iloc[:, 1:].values
    pm_data = data[data[0] == 'PM'].iloc[:, 1:].values

    N = 50

    # Concatenate all PM rows and then compute a single histogram
    pm_aggregated = pm_data.ravel()
    pm_histogram = np.histogram(pm_aggregated, bins=N)[0]
    pm_bins = np.histogram(pm_aggregated, bins=N)[1]

    # Compute histograms for MP (assuming 10 bins for simplicity)
    mp_histograms = [np.histogram(row, bins=pm_bins)[0] for row in mp_data]



    # plt.plot(pm_histogram[1:], label="PM")
    # # sprint (np.mean(pm_data.ravel()))
    # for counter, mp_hist in enumerate(mp_histograms):
    #     # sprint(np.mean(mp_data[counter, 1:].ravel()))
    #     plt.plot(mp_hist[1:], label="MP"+ str(counter + 1), alpha=(counter+1)/28, color="red")
    # plt.legend()
    # plt.xlim(1, N)
    # sprint (np.histogram(pm_aggregated, bins=N)[1])
    # plt.xticks(np.histogram(pm_aggregated, bins=N)[1])
    # sprint (np.histogram(pm_aggregated, bins=N)[0])
    # plt.yscale("symlog")
    # plt.savefig("actual_hists.png")

    # Compute the distance between each MP histogram and the PM histogram
    # Here, we use the Euclidean distance
    # distances = [np.linalg.norm(mp_hist[1:] - pm_histogram[1:]) for mp_hist in mp_histograms]

    distances = [wasserstein_distance(mp_hist[1:], pm_histogram[1:]) for mp_hist in mp_histograms]


    # sprint (len(distances), val_loss.shape)
    # plt.scatter(distances, val_loss, label=r"$p_h=$" + P_H)

    plt.scatter(distances[argmin], val_loss[argmin], label=r"$p_h=$" + P_H + " min val MSE")

    sprint (pearsonr(distances, val_loss), spearmanr(distances, val_loss))
    # plt.clf()
    # plt.plot(distances)
plt.xlabel("Wasserstein |IC-MC|")
plt.ylabel("Val MSE")
plt.legend()
plt.tight_layout()
plt.savefig("histcompare_different_PH_val_loss_min_val.png")
