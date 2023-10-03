from smartprint import smartprint as sprint
import numpy as np


class NaiveBaseline:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.naive_baseline_mse = np.mean(((x - y) ** 2).flatten())

    # @classmethod
    # def from_dataloader(cls, data_loader, N):
    #     """
    #     data_loader: each call returns x, y
    #     N: How many batches to check
    #     """
    #     counter = 0
    #     val = []
    #     sprint (data_loader)
    #     for x, y in data_loader:
    #         if counter >= N:
    #             break
    #         val.append(cls(x, y).naive_baseline_mse)
    #     cls(x,y).naive_baseline_mse = np.mean(val)

    def from_dataloader(self, data_loader, N):
        """
        data_loader: each call returns x, y
        N: How many batches to check
        """
        counter = 0
        val = []
        val_non_zero = []
        # sprint (data_loader)
        for values in data_loader:
            if len(values) == 3:
                X, Y, _ = values
            elif len(values) == 2:
                X, Y = values
            else:
                raise ValueError("Unexpected number of values returned by dataloader")

            # sprint(X.shape, Y.shape)
            x = X
            y = Y

            # repeat the channels manually
            for i in range(X.shape[1]):
                x[:, i, :, :, :] = X[:, -1, :, :, :]

            # Broadcasting does not seem to work
            # even with just the last value
            # :(; so we can just compare the last frame repeated manually
            # x = np.moveaxis(x, -1, 1)
            # y = np.moveaxis(y, -1, 1)

            # sprint(x.shape, y.shape)
            if counter >= N:
                break
            val.append(np.mean(((x - y) ** 2).flatten()))
            val_non_zero.append(np.mean(((x[x > 0] - y[x > 0]) ** 2).flatten()))
        self.naive_baseline_mse = np.mean(val)
        self.naive_baseline_mse_non_zero = np.mean(val_non_zero)
        return self
