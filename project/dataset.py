import numpy as np
from dataclasses import dataclass
from utils import Config
from typing import List, Generator, Tuple
import pickle
import os


# Not sure if we want to group stuff like this into sample
@dataclass(frozen=True)
class DataSample:
    index: int
    csi: np.ndarray
    loc: np.ndarray
    vel: np.ndarray
    ant_orient: np.ndarray


@dataclass(frozen=True)
class DataWindow:
    sample: DataSample  # Data Sample at index
    csi_window: np.ndarray
    loc_window: np.ndarray
    vel_window: np.ndarray
    orient_window: np.ndarray


def create_windows_from(data: np.ndarray, window_indexes: np.ndarray, window_size: int):
    """
    :param data: N by <any dim>
    :param window_indexes: Indexes of the start of the window. These indexes are not included in the output, only the previous window_size are
    :param window_size: The size of each window
    :return: N by window_size by <any dim>
    """
    assert np.all(window_indexes >= window_size)

    output_shape = (len(window_indexes), window_size) + data.shape[1:]
    output = np.zeros(output_shape, dtype=data.dtype)

    for i in range(len(window_indexes)):
        idx = window_indexes[i]
        output[i] = data[idx - window_size:idx]

    return output


class Dataset:
    csi_samples: np.ndarray
    ue_locations: np.ndarray
    ue_velocity: np.ndarray
    antenna_orientations: np.ndarray

    def __init__(self, cfg: Config, csis: np.ndarray, ue_locs: np.ndarray, ue_vels: np.ndarray, antenna_orientations: np.ndarray, window_indexes: np.ndarray):
        """
        :param csis:            Points to entire array of CSIs
        :param ue_locs:         Points to entire array of Locs
        :param ue_vels:         Points to entire array of Vels
        :param antenna_orientations:       Points to entire array of Antenna Orientations
        :param window_indexes:  List of indices of end of prediction Window.
            ie for each index, the previous cfg.predictor_window_size values will be used to predict the CSI at that index.

        """
        self.cfg = cfg
        self.csi_samples = csis[window_indexes]
        self.ue_locations = ue_locs[window_indexes]
        self.ue_velocity = ue_vels[window_indexes]
        self.antenna_orientations = antenna_orientations[window_indexes]

        self.csi_windows = create_windows_from(csis, window_indexes, cfg.predictor_window_size)
        self.loc_windows = create_windows_from(ue_locs, window_indexes, cfg.predictor_window_size)
        self.vel_windows = create_windows_from(ue_vels, window_indexes, cfg.predictor_window_size)
        self.orient_windows = create_windows_from(antenna_orientations, window_indexes, cfg.predictor_window_size)

    def __getitem__(self, index) -> DataSample:
        return DataSample(index, self.csi_samples[index], self.ue_locations[index], self.ue_velocity[index], self.antenna_orientations)

    def __len__(self):
        return len(self.csi_samples)

    def __iter__(self):
        # Iterate over Samples
        for i in range(len(self)):
            yield self[i]

    def get_window(self, index) -> DataWindow:
        return DataWindow(
            self[index],
            self.csi_windows[index],
            self.loc_windows[index],
            self.vel_windows[index],
            self.orient_windows[index]
        )

    def each_window(self) -> Generator[DataWindow, None, None]:
        # Iterate over each Window
        for i in range(len(self)):
            yield self.get_window(i)

    # Increase the length of the training data
    def duplicate_and_add_noise(self, times, noise_proportion):
        # TODO: Use SNR as in reference.ipynb?
        data_fields = [
            "csi_samples",
            "ue_locations",
            "ue_velocity",
            "antenna_orientations",
            "csi_windows",
            "loc_windows",
            "vel_windows",
            "orient_windows",
        ]
        for name in data_fields:
            curr = getattr(self, name)
            duplicated_data = np.repeat(curr, times, axis=0)
            var = np.var(curr)
            if noise_proportion > 0:
                noise = np.random.normal(0, var * noise_proportion, duplicated_data.shape)
                if np.any(np.iscomplex(curr)):
                    noise = noise + 1j * np.random.normal(0, var * noise_proportion, duplicated_data.shape)
                duplicated_data = duplicated_data + noise
            setattr(self, name, duplicated_data)


def load_data(cfg: Config) -> Tuple[Dataset, Dataset]:
    with open(cfg.data_path, "rb") as f:  # change file link for your machine
        data = pickle.load(f)

    freq_channel = np.array(data["freq_channel"])
    freq_channel = np.squeeze(freq_channel)
    ue_loc = np.array(data["UE_loc"])
    ue_speed = np.array(data["UE_speed"])
    antenna_orient = np.array(data["antenna_orient"])

    indices = np.random.permutation(freq_channel.shape[0] - cfg.predictor_window_size) + cfg.predictor_window_size
    # Don't take the beginning window_size because then the windows will be too small.

    split_index = int(len(indices) * cfg.train_test_split)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_set = Dataset(cfg, freq_channel, ue_loc, ue_speed, antenna_orient, train_indices)
    test_set = Dataset(cfg, freq_channel, ue_loc, ue_speed, antenna_orient, test_indices)

    train_set.duplicate_and_add_noise(cfg.duplicate_data, 1.0/cfg.train_snr)
    test_set.duplicate_and_add_noise(cfg.duplicate_data, 1.0/cfg.test_snr)

    return train_set, test_set


def dataset_from_path(path, cfg) -> Dataset:
    with open(path, "rb") as f:  # change file link for your machine
        data = pickle.load(f)

    freq_channel = np.array(data["freq_channel"])
    freq_channel = np.squeeze(freq_channel)
    ue_loc = np.array(data["UE_loc"])
    ue_speed = np.array(data["UE_speed"])
    antenna_orient = np.array(data["antenna_orient"])

    # indices = np.arange(freq_channel.shape[0])
    indices = np.arange(freq_channel.shape[0] - cfg.predictor_window_size) + cfg.predictor_window_size
    # Don't take the beginning window_size because then the windows will be too small.

    train_set = Dataset(cfg, freq_channel, ue_loc, ue_speed, antenna_orient, indices)

    return train_set


def combine_time_series_paths(directory,path_indices,desired_freq,save_filename="train.pickle"):
    list_of_dicts = []
    if not os.path.isdir(directory):
        raise ValueError(f"The specified directory does not exist: {directory}")

    channels = []
    locations = []
    speeds = []
    ant_orients = []

    is_reverse = False
    for filename in os.listdir(directory):
        # Check if the file is a pickle file
        if "fc" in filename:
            # Construct full file path
            filepath = os.path.join(directory, filename)

            path_ind = int(filename.split("_")[0][0:-4])
            freq = int(filename.split("_")[6].split(".")[0])

            if(path_ind in path_indices and freq == desired_freq):

                with open(filepath, 'rb') as f:
                    # Store the contents in the dictionary, using filename as the key
                    data = pickle.load(f)

                freq_channel = np.array(data["freq_channel"])
                # freq_channel = np.squeeze(freq_channel)
                ue_loc = np.array(data["UE_loc"])
                ue_speed = np.array(data["UE_speed"])
                antenna_orient = np.array(data["antenna_orient"])

                if(is_reverse):
                    channels.append(freq_channel[::-1])
                    locations.append(ue_loc[::-1])
                    speeds.append(ue_speed[::-1])
                    ant_orients.append(antenna_orient[::-1])
                    is_reverse = False
                else:
                    channels.append(freq_channel)
                    locations.append(ue_loc)
                    speeds.append(ue_speed)
                    ant_orients.append(antenna_orient)
                    is_reverse = True

    channels = np.vstack(channels)
    speeds = np.vstack(speeds)
    ant_orients = np.vstack(ant_orients)
    locations = np.vstack(locations)

    data = {}
    data["freq_channel"] = channels
    data["UE_loc"] = locations
    data["UE_speed"] = speeds
    data["antenna_orient"] = ant_orients

    print("Saving files to ",os.path.join(directory, save_filename))
    print("Number of samples",data["freq_channel"].shape)
    with open(os.path.join(directory, save_filename), "wb") as f:
        pickle.dump(data, f)

