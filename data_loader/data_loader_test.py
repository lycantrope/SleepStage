from data_loader.data_loaders import SleepDataLoader
from pathlib import Path

homedir = Path("/Users/chungkuanchen/Projects/MNE_opencv")
dataloader = SleepDataLoader(homedir, 10, num_workers=1)


if __name__ == "__main__":
    k = 10
    for data in dataloader:
        print(data)
        k -= 1
        if k == 0:
            break
