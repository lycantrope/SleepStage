from pathlib import Path
import torch
import h5py
from torch.utils.data import Dataset
import json
from scipy import signal


class BasicSleepDataset(Dataset):
    def __init__(
        self, datadir, window_size: int = 4, useFFT: bool = True, useSTFT: bool = True
    ):
        self.datadir = Path(datadir)
        self.h5dict = {
            i: {"path": file, "EEG_header": self.loadH5header(file, "EEG")}
            for i, file in enumerate(sorted(self.datadir.glob("*.h5")))
        }
        assert len(self.h5dict) > 0, "Cannot found any h5 file"
        self.useFFT = useFFT
        self.useSTFT = useSTFT
        self.window_size = window_size
        self.epochs = self.__get_lens()

    def loadH5header(self, path, filter=None):
        tempdict = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                tempdict[key] = json.loads(f[filter].attrs["signal_headers"])
        if isinstance(filter, str):
            return tempdict.get(filter, None)
        else:
            return tempdict

    def __get_lens(self):
        totalepochs = 0
        for key, value in self.h5dict.items():
            value["epochs"] = value["EEG_header"]["sample_size"] / (
                self.window_size * value["EEG_header"]["sample_rate"]
            )
            self.h5dict[key] = value
            totalepochs += value["epochs"]
        return int(totalepochs)

    def __len__(self):
        return self.epochs

    def __repr__(self):
        fmt = "{0}:\t{1}"
        msgstrs = [
            fmt.format(key, val["path"].name) for key, val in self.h5dict.items()
        ]
        return "\n".join(msgstrs)

    def __getitem__(self, id):
        for i in range(len(self.h5dict)):
            count = self.h5dict[i]["epochs"]
            file = self.h5dict[i]["path"]
            if id < count:
                break
            id -= count
        id = int(id)
        output = {}
        with h5py.File(file, "r") as f:
            dset = f.get("EEG")
            assert dset is not None, f"{file.name} file did not contain EEG data"
            header = json.loads(dset.attrs["signal_headers"])
            window_size = header["window_size"]
            sample_rate = header["sample_rate"]
            preFFT = header["preFFT"]
            preSTFT = header["preSTFT"]
            start = int(id * self.window_size * sample_rate)
            end = int((id + 1) * self.window_size * sample_rate)
            rawdata = dset[0, start:end]
            output["EEG"] = torch.from_numpy(rawdata).type(torch.FloatTensor)
            # get labels
            dset = f.get("Labels")
            assert dset is not None, f"{file.name} file did not contain labels"
            output["Labels"] = torch.from_numpy(dset[id]).type(torch.LongTensor)
            dset = f.get("daytime")
            assert dset is not None, f"{file.name} did not contain dayteim"
            output["daytime"] = torch.from_numpy(dset[id]).type(torch.LongTensor)

            output["FFT"] = None
            output["STFT"] = None
            if self.useFFT:
                if (window_size == self.window_size) and preFFT:
                    dset = f.get("EEG_FFT")
                    fft = dset[0, id]
                else:
                    _f, fft = signal.welch(
                        rawdata,
                        fs=sample_rate,
                        nperseg=sample_rate * self.window_size / 2,
                    )
                output["FFT"] = torch.from_numpy(fft.real).type(torch.FloatTensor)

            if self.useSTFT:
                if (window_size == self.window_size) and preSTFT:
                    dset = f.get("EEG_STFT")
                    stft = dset[0, id]
                else:
                    _f, t, stft = signal.stft(
                        rawdata,
                        fs=sample_rate,
                        nperseg=sample_rate * self.window_size / 2,
                    )
                output["STFT"] = torch.from_numpy(stft.real).type(torch.FloatTensor)
        return output
