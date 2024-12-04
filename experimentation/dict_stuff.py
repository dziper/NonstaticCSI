import os
import pickle
import numpy as np
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


directory = f"C:\\Users\ibrahimkilinc\Documents\ECE257_Project\\NonstaticCSI\Dataset"
desired_fc = 2.5e9
train_dir = f"train_series.pickle"
train_indices = [i for i in range(33)]

test_indices = [i for i in range(34,40)]
combine_time_series_paths(directory,train_indices, desired_fc, train_dir)

test_dir = f"test_series.pickle"
combine_time_series_paths(directory,test_indices, desired_fc, test_dir)