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


if __name__ == "__main__":
    # os.system("rm -rf data_samples && mkdir data_samples")
    plt.clf()
    for city_count, city in enumerate(config.city_list):
        # scales
        for scale_count, scale in enumerate([25, 55, 105]):
            if city_count == 2:
                plt.rc("xtick", labelsize=6)
            else:
                plt.rc("xtick", labelsize=0)

            if scale_count == 0:
                plt.rc("ytick", labelsize=6)
            else:
                plt.rc("ytick", labelsize=0)

            for i_o_length in config.i_o_lengths_def:
                for pred_horiz in config.pred_horiz_def:
                    obj = ProcessRaw(
                        cityname=city, i_o_length=i_o_length, prediction_horizon=pred_horiz, grid_size=scale
                    )
                    x = 0
                    for n in tqdm(range(1, 101), desc="loading city files" + obj.key_dimensions()):
                        x += np.load(os.path.join(obj.training_folder, obj.key_dimensions() + str(n) + "_x.npy"))
                    x = np.mean(x, axis=-1)
                    assert x.shape == (scale, scale)

                    plt.subplot(3, 3, city_count * 3 + scale_count + 1)

                    plt.gca().set_title(obj.key_dimensions(), fontsize=6)

                    plt.imshow(x, vmin=0, vmax=5000, cmap="jet", aspect=1, interpolation="none")

    # reset font size before colorbar
    plt.rc("ytick", labelsize=6)
    plt.rc("xtick", labelsize=6)
    plt.colorbar()
    plt.savefig(os.path.join("city_plots", "all_cityplots" + ".png"), dpi=600)
    # plt.show()
