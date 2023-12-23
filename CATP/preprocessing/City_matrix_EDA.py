import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # location of config file

import pathlib
import datetime
import glob
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.patches import Rectangle
from tqdm import tqdm
import datetime
from smartprint import smartprint as sprint
from slugify import slugify
import matplotlib
matplotlib.use('TkAgg')  # You can replace 'TkAgg' with 'Agg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt

sys.path.append(config.DATA_FOLDER)  # location of t4c22 folder
import t4c22
from t4c22.t4c22_config import load_basedir

from haversine import haversine, Unit
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.patches import Rectangle
import t4c22
from t4c22.t4c22_config import load_basedir
from preprocessing.ProcessRaw import ProcessRaw

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Assuming ProcessRaw and config are defined elsewhere in your code

# Define the aspect ratios for each city
city_aspects = {
    'London': 30.04 / 49.16,  # Width / Height
    'Melbourne': 38.37 / 54.65,
    'Madrid': 21.67 / 20.38
}

if __name__ == "__main__":
    plt.clf()
    for city_count, city in enumerate(config.city_list):
        aspect_ratio = city_aspects[city.title()] # Default to 1 if city not in dictionary

        for scale_count, scale in enumerate([25, 55, 105]):#, 55, 105]):  # Extend this list as needed
            plt.rc("xtick", labelsize=6 if city_count == 2 else 0)
            plt.rc("ytick", labelsize=6 if scale_count == 0 else 0)

            for i_o_length in [1]: # config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    obj = ProcessRaw(
                        cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                    )
                    x = 0
                    for n in tqdm(range(1, 101), desc="loading city files" + obj.key_dimensions()):
                        x += np.load(os.path.join(obj.training_folder, obj.key_dimensions() + str(n) + "_x.npy"))
                    x = np.mean(x, axis=-1)
                    assert x.shape == (scale, scale)

                    ax = plt.subplot(3, 3, city_count * 3 + scale_count + 1)
                    im = plt.imshow(x, vmin=0, vmax=5000, cmap="jet", aspect=aspect_ratio, interpolation="none")
                    plt.gca().set_title(r"$Task( i_o=$"+str(i_o_length) +r", $p_h=$" + str(pred_horiz) + r", $s=$"+str(scale) + r"$)$", fontsize=6)

                    # Adding colorbar to each subplot
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.ax.tick_params(labelsize=9)

    # Save the figure
    plt.savefig(os.path.join("city_plots", "all_cityplots" + ".png"), dpi=600)
    # plt.show()
