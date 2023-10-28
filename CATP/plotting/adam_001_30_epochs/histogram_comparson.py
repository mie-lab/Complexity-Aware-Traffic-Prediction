import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
from smartprint import smartprint as sprint

# Load the CSV data
data = pd.read_csv('record_distribution.csv', header=None)

# Separate the data based on MP and PM markers
mp_data = data[data[0] == 'MP'].iloc[:, 1:].values
pm_data = data[data[0] == 'PM'].iloc[:, 1:].values

N = 10

# Concatenate all PM rows and then compute a single histogram
pm_aggregated = pm_data.ravel()
pm_histogram = np.histogram(pm_aggregated, bins=N)[0]
pm_bins = np.histogram(pm_aggregated, bins=N)[1]

# Compute histograms for MP (assuming 10 bins for simplicity)
mp_histograms = [np.histogram(row, bins=pm_bins)[0] for row in mp_data]



plt.plot(pm_histogram[1:], label="PM")
sprint (np.mean(pm_data.ravel()))
for counter, mp_hist in enumerate(mp_histograms):
    sprint(np.mean(mp_data[counter, 1:].ravel()))
    plt.plot(mp_hist[1:], label="MP"+ str(counter + 1), alpha=(counter+1)/28, color="red")
plt.legend()
plt.xlim(1, N)
# sprint (np.histogram(pm_aggregated, bins=N)[1])
# plt.xticks(np.histogram(pm_aggregated, bins=N)[1])
# sprint (np.histogram(pm_aggregated, bins=N)[0])
# plt.yscale("symlog")
plt.savefig("actual_hists.png")

# Compute the distance between each MP histogram and the PM histogram
# Here, we use the Euclidean distance
distances = [np.linalg.norm(mp_hist[1:] - pm_histogram[1:]) for mp_hist in mp_histograms]

print(distances)

plt.clf()
plt.plot(distances)
plt.savefig("histcompare.png")
