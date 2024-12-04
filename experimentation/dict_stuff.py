import os
import pickle
import numpy as np
def load_pickle_files(directory):
    list_of_dicts = []
    if not os.path.isdir(directory):
        raise ValueError(f"The specified directory does not exist: {directory}")
    for filename in os.listdir(directory):
        # Check if the file is a pickle file
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            # Construct full file path
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                # Store the contents in the dictionary, using filename as the key
                data = pickle.load(f)
                freq_channel = np.array(data["freq_channel"])
                freq_channel = np.squeeze(freq_channel)
                ue_loc = np.array(data["UE_loc"])
                ue_speed = np.array(data["UE_speed"])
                antenna_orient = np.array(data["antenna_orient"])
                pickle_data = {"CSI": freq_channel, "Location": ue_loc, "Speed": ue_speed, "Antenna Orientation": antenna_orient}
                list_of_dicts.append(pickle_data)
    return list_of_dicts

directory_path = r"C:\Users\nrazavi\Downloads\drive-download-20241204T004446Z-001"

loaded_pickles = load_pickle_files(directory_path)