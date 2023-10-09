import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import numpy as np
from smartprint import smartprint as sprint

# Define the CSV filenames
files = {
    "validation-create_model_dummy_single_layer-sgd-0001-25_madrid_.csv": "f_small_madrid",
    "validation-create_model_small_epochs-sgd-0001-25_madrid_.csv": "f_mid_madrid",
    "validation-create_model_f_def_no_BN-sgd-0001-25_madrid_.csv": "f_big_madrid",

    "validation-create_model_dummy_single_layer-sgd-0001-london-25_.csv": "f_small_london",
    "validation-create_model_small_epochs-sgd-0001-london-25_.csv": "f_mid_london",
    "validation-create_model_f_def_no_BN-sgd-0001-london-25_.csv": "f_big_london",
}

for FILE in files:


    alphas = [1, 0.5, 0.25, 0.2,1,1,1,1] # , 0.1, 0.5]

    columns = [
                'epoch',
                 # 'CSR_GB_sum',
                 # 'CSR_MP_count',
                 # 'CSR_MP_std',
                 # 'CSR_MP_sum',
                 # 'CSR_NM_sum',
                 # 'CSR_PM_std',
                 # 'CSR_PM_sum',
                 'loss',
                 # 'naive-model-mse',
                 # 'naive-model-non-zero',
                 # 'non_zero_mse',
                 # 'self.CSR_GB_count',
                 # 'self.CSR_NM_count',
                 # 'self.CSR_PM_count',
                 'val_loss',
                 'val_non_zero_mse',
                    "MC",
                    "IC",
                    # "val/MC"
                ]




    color_dict = {column: mcolors.to_hex([random.random(), random.random(), random.random()]) for column in columns}
    color_dict["IC"] = 'green'
    color_dict["MC"] = 'red'
    color_dict['val_loss'] = 'blue'

    plt.clf()
    # Loop through the CSV files
    # for idx, file in enumerate(files.keys()):
    if 2==2:

        file = FILE
        idx = 0

        # Load the data
        data = pd.read_csv(file)

        # Plot each column on the same plot
        n = 3


        # data["val/train"] = data["val_non_zero_mse"] / data["non_zero_mse"]
        data["MC"] = data["CSR_MP_sum"]
        data["IC"] = data["CSR_PM_sum"]
        data["IC"] = data["IC"].max()  # To take care of avoiding recomputation


        linestyle = '-'
        # if "create_model_f_big" in file:
        #     linestyle = ":"
        for col in columns:

            if idx >= 0 :
                # Exclude 'epoch' column from plotting

                if col != 'epoch':

                    if "validation-create_model.csv" in files:
                        scale = 10000
                    else:
                        scale = 5000


                    if col in ["MC"]:
                        data[col] = data[col] * scale
                    if col in ["IC"]:
                        data[col] = data[col] * 1 # Since we don't use any Data loader in IC computation
                    elif col in ["val_loss", "loss", "val_non_zero_mse", "non_zero_mse", "naive-model-mse","naive-model-non-zero"]:
                        data[col] = data[col] * scale * scale
                        # continue
                    # max_ = data[col].max()
                    # data[col] = data[col] / max_

                    plt.plot(data['epoch'], np.convolve(data[col], [1/n]*n, "same"),
                             alpha=alphas[idx], color=color_dict[col], label=col + "_" + str(files[file]),
                             linestyle=linestyle)






    plt.title('MC evaluation during training', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(fontsize=6, ncol=2, loc="upper right")
    plt.xticks(list(range(0, 50, 1)), rotation=90, fontsize=4)
    # plt.yscale("log")
    plt.grid(axis='x',alpha=0.05)
    plt.xlim(1, 50-n)
    # plt.ylim(1, 4000)

    plt.savefig("three_models_scale_25_" +str(files[FILE])+ ".png", dpi=300)
    # plt.show()




    plt.clf()
    min_index = np.argmin(data["val_loss"])
    best_MC = data["MC"][min_index]
    # sprint (best_MC)


    loss_diff = np.abs(data["val_loss"] - data["loss"])
    mc_diff = np.abs(data["MC"] - data["IC"])
    where_ = mc_diff > (data["IC"] - best_MC)

    # loss_diff = loss_diff[where_]
    # mc_diff = mc_diff[where_]


    from scipy import stats

    print (files[FILE])
    sprint (stats.pearsonr(loss_diff, mc_diff**2))
    sprint (data["val_loss"].min(), data["val_non_zero_mse"].min())

    plt.scatter(loss_diff, mc_diff**2, s=0.8)
    plt.xlabel('loss_diff')
    plt.ylabel("mc_diff")
    plt.savefig("loss_diff_vs_mc_diff_f_big.png", dpi=300)
    # plt.show()