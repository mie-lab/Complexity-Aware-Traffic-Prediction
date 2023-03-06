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
        # sprint (data_loader)
        for x, y in data_loader:
            # sprint (x.shape, y.shape)
            if counter >= N:
                break
            val.append(np.mean(((x - y) ** 2).flatten()))
        self.naive_baseline_mse = np.mean(val)
        return self
