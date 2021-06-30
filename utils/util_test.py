from pathlib import Path
from utils.util import *
import h5py
import json

homedir = Path("/Users/chungkuanchen/Projects/MNE_opencv")
edfpath = Path("/Users/chungkuanchen/Projects/MNE_opencv/test.edf")
csvlist = sorted(homedir.glob("*.csv"))
# signals, signal_headers, header = highlevel.read_edf(str(edfpath))
convert_edf_file(edfpath, 4, csvlist)


# freq, headsig, head = highlevel.read_edf(str(edfpath), ch_names= "EEG1-2", verbose=True)
h5list = sorted(homedir.glob("*.h5"))

# file = h5py.File(h5list[1], "r")
for path in h5list:
    with h5py.File(path, "r") as f:
        for key in f.keys():
            print(f[key])
