from base import BaseDataLoader
from data_loader.dataset import BasicSleepDataset


class SleepDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.data_dir = data_dir
        self.dataset = BasicSleepDataset(self.data_dir)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
