import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('agg')

# Function to calculate histogram envelope
def get_histogram_envelope(values, bins=50):
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

# Read and plot each line from the files
def plot_file(file_name, label, color, linestyle):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Calculate alpha values
    alphas = np.linspace(0.03, 0.9, len(lines))

    for idx, line in enumerate(lines[-10:]):
        # Filter values within the range 0 to 2000
        values = [float(val) for val in line.strip().split(',')[1:] if 0 <= float(val) <= 2000]

        # Get histogram envelope
        bin_centers, counts = get_histogram_envelope(values)
        plt.plot(bin_centers, counts, label=label if idx == 0 else "", color=color, linestyle=linestyle, alpha=alphas[idx])

# File names and corresponding labels and colors for BN 1 and No BN 1
file_info = [
    ("BN_1_log.csv", "BN 1", "green", '-'),
    ("No_BN_1_log.csv", "No BN 1", "blue", '--')
]

# Plot each distribution
for file_name, label, color, linestyle in file_info:
    plot_file(file_name, label, color, linestyle)

plt.title('Distribution of Values (0 to 2000)')
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.yscale("log")
plt.ylim(0, 200)
plt.xlim(500, 2000)
plt.legend()
plt.savefig("Compare_BN_no_BN_CRIT_hist.png", dpi=300)
plt.show()
