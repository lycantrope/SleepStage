import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from re import T

import h5py
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from scipy import signal
from pyedflib import highlevel


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# Type of stage to be characterized.
__STAGES = ["W", "NR", "R"]


def convert_edf_file(
    edfpath: Path,
    numsofsamples: int,
    csvlist: list,
    windowsize: int = 4,
    normalization=True,
    saveEMG=False,
    saveLOC=False,
    lighton=9,
    calculate_fft: bool = False,
    calculate_stft: bool = False,
):
    """
    Take the edffile and split each individual mice into individual hdf.
    [OPTION] processing the FFT spontaniously. (Welch)
    [OPTION] processing STFT. (STFT)
    Parameters
    ------
    edfpath: the path to edf file
    numsofsamples: the num of mice in edffiles
    windowsize: the window size (sec)
    convert_to_fft:
    """
    edfpath = Path(edfpath)
    csvlist = sorted(map(Path, csvlist))
    csvdata = {i: (file.name, read_stages(file)) for i, file in enumerate(csvlist)}
    assert numsofsamples == len(
        csvlist
    ), "The sample number should have same size of label csv"
    assert len(csvdata) == numsofsamples, "Fail to read the csvdata"
    assert (
        edfpath.is_file() and edfpath.suffix == ".edf"
    ), "File does not exist or is not a edf file"
    headers = highlevel.read_edf_header(str(edfpath))
    starttime = headers["startdate"]
    duration = headers["Duration"]
    endtime = starttime + timedelta(seconds=int(duration))
    headers["startdate"] = str(starttime)
    headers["endtime"] = str(endtime)
    write_json(headers, edfpath.with_suffix(".json"))
    channels = np.array(headers["channels"]).reshape(numsofsamples, -1)
    LDtime = getLDtime(starttime, endtime, lighton, windowsize)
    for i, chs in enumerate(channels):
        print(f"Saving...: HDF file({i+1}/{numsofsamples})", flush=True)
        h5path = edfpath.parent.joinpath(edfpath.stem + f"_{i+1}.h5")
        with h5py.File(h5path, mode="w") as f:
            # label
            dset = f.create_dataset("Labels", data=csvdata[i][1], dtype="?")
            dset.attrs["filename"] = csvdata[i][0]

            # daytime
            f.create_dataset(
                "daytime", data=LDtime[: csvdata[i][1].shape[0]], dtype="?"
            )
            # chs
            for label in chs:
                if not saveEMG and "EMG" in label:
                    continue
                if not saveLOC and "LOC" in label:
                    continue
                rawdata, header, _ = highlevel.read_edf(
                    str(edfpath), ch_names=label, verbose=True
                )
                header = header[0]
                chunk = int(header["sample_rate"] * windowsize)
                rawdata = rawdata[:, : csvdata[i][1].shape[0] * chunk]
                header["Duration"] = duration
                header["window_size"] = windowsize
                header["sample_size"] = rawdata.size
                header["preFFT"] = False
                header["preSTFT"] = False
                label = label.rstrip("0123456789-")
                if normalization:
                    mu, sigma = rawdata.mean(), rawdata.std()
                    rawdata = (rawdata - mu) / sigma
                dset = f.create_dataset(label, data=rawdata, dtype="f8")
                dset.attrs["signal_headers"] = json.dumps(header)
                if label == "EEG":
                    if calculate_fft:
                        header["preFFT"] = True
                        print("calculate FFT")
                        _f, pxx = signal.welch(
                            rawdata.reshape(chunk, -1), fs=header["sample_rate"]
                        )
                        f.create_dataset(label + "_FFT", data=pxx.real, dtype="f8")
                    if calculate_stft:
                        header["preSTFT"] = True
                        print("calculate STFT")
                        _f, t, Zxx = signal.stft(
                            rawdata.reshape(chunk, -1), fs=header["sample_rate"]
                        )
                        dset = f.create_dataset(
                            label + "_STFT", data=Zxx.real, dtype="f8"
                        )
    print("Finished")


def get_rows(path):
    MAX_SEARCH_LINES = 100
    with open(path, "r") as file:
        for i in range(MAX_SEARCH_LINES):
            line = file.readline()
            if "EEG" in line:
                return i + 1
    print("The output file from SleepSign was wrong!")
    return None


def read_stages(csvpath):
    csvpath = Path(csvpath)
    assert (
        csvpath.is_file() and csvpath.suffix == ".csv"
    ), "File does not exist or is not a csv file"
    skiprow = get_rows(csvpath)
    df = pd.read_csv(csvpath, skiprows=skiprow)
    for s in __STAGES:
        df[s] = df["Stage"] == s
    return df[__STAGES].astype("?").to_numpy()


def readHDF(h5path, label, id=None):
    with h5py.File(h5path, "r") as file:
        if label not in file.keys():
            return None
        else:
            if id == "header":
                return json.loads(file[label].attrs["signal_headers"])
            elif id is None:
                return file[label][()]
            else:
                return file[label][id]


def getLDtime(start_time, end_time, lighton, window_size):
    timepoints = [start_time]
    t = datetime(start_time.year, start_time.month, start_time.day - 1, hour=lighton)
    while (end_time - t).days >= 0:
        if (t - start_time).days >= 0:
            timepoints.append(t)
        t = t + timedelta(hours=12.0)
    timepoints.append(end_time)
    slices = getTimeSlice(timepoints)
    arr = getLDArray(slices)
    if (lighton + 12) >= t.hour and (t.hour >= lighton):
        if not arr[0]:
            arr = ~arr
    else:
        if arr[0]:
            arr = ~arr
    return np.expand_dims(arr[::window_size], 1)


def getDeltaSecond(t0, t1):
    d = abs(t0 - t1)
    return d.days * 24 * 60 * 60 + d.seconds


def getTimeSlice(timepoints):
    timepoints = sorted(timepoints)
    return list(map(lambda x: getDeltaSecond(x, timepoints[0]), timepoints))


def getLDArray(slice):
    slice = sorted(slice)
    duration = slice[-1] - slice[0]
    arr = np.zeros((duration), dtype=bool)
    for i in range(len(slice) - 1):
        t0 = slice[i]
        t1 = slice[i + 1]
        arr[t0:t1] = True
        arr = ~arr
    return arr
