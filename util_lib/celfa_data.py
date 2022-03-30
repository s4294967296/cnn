# STL
import os
import logging
import configparser
import pickle
import pathlib
import random

# 3rd party
import numpy as np

# Home-made :)
import celfa_exceptions

########################################################################################################################
# CELFA - CNN Evaluation Library For ANNIE
# ----------------------------------------

# This file encapsulates utility functions to load and organize data.
# In the first parts of the file, basic file operations are implemented, in the latter parts building the data
# structures and export, as well as saving is implemented.
########################################################################################################################

########################################################################################################################
# CONFIG:
from typing import List

config_data = configparser.ConfigParser()
config_data.read("data_config.ini")

########################################################################################################################
# LOGGING:

data_log_formatter = logging.Formatter(config_data["LOGGER_DATA"]["format"])

data_log_file_handler = logging.FileHandler(config_data["LOGGER_DATA"]["file"])
data_log_file_handler.setFormatter(data_log_formatter)

data_logger = logging.getLogger(__name__)
data_logger.addHandler(data_log_file_handler)
data_logger.setLevel(logging.INFO)

########################################################################################################################


def find_number_files(path: str, label: str, name: str, ending: str = "csv", guess: int = -1) -> int:
    """
    This function returns the amount of files fount, specified by the following parameters in the form
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number. The function will start attempting
    to load files starting with the index '0'.

    Warning: This will only find all files, as long as the highest file index is smaller than the number of all files
    in the directory.

    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param name: What data is read in, e.g. 'charge', 'neutron number'
    :param ending: File ending, e.g. 'csv' (without dot). Default = 'csv'
    :param guess: Best guess of number of data files. If set to -1, attempt to find all files fitting the structure.
     Default = -1.
    :return: Returns how many files, specified by the parameters, have been found.
    """
    try:
        os.listdir(path)
    except OSError:
        data_logger.critical(f"OSErr: Call to find_number_files: path ({path}) does not lead to valid directory")
        return -1

    counter = guess if guess != -1 else int(len(os.listdir(path)))
    data_logger.info(f"Call to find_number_files ({path}{label}_###_{name}.{ending}): files found in directory: "
                     f"{counter}")

    for i in range(counter):
        p = pathlib.Path(f"{path}{label}_{i}_{name}.{ending}")
        if not p.is_file():
            counter -= 1
            data_logger.debug(f"Call to find_number_files - {path}{label}_{i}_{name}.{ending}: file not found")
            continue

    data_logger.info(f"Call to find_number_files ({path}{label}_###_{name}.{ending}): valid files found in directory: "
                     f"{counter}")
    return counter


def find_number_events(path: str, label: str, name: str, ending: str = "csv",
                       guess: int = -1) -> int:
    """
    Check all available files, return the amount of single events found in the files
    specified by the parameters of the form
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number. Start attempting
    to load files beginning with the index '0'.

    Warning: This will only find all events in all files, as long as the highest file index is smaller than the number
    of all files in the directory.

    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param name: What data is read in, e.g. 'charge', 'neutron number'
    :param ending: File ending, e.g. 'csv' (without dot). Default = 'csv'
    :param guess: Best guess of number of data files. If guess = -1, the function will attempt
    to load all files following the structure above. Default = -1 (will try at most #guess files)
    :return: Returns how many events (that is: entries separated by the delimiter ',' inside the loaded files) have
    been found.
    """
    event_counter = 0
    try:
        os.listdir(path)
    except OSError:
        data_logger.critical(f"OSErr: Call to find_number_events: path ({path}) does not lead to valid directory")
        return -1

    file_counter = guess if guess != -1 else int(len(os.listdir(path)))
    data_logger.info(f"Call to find_number_events ({path}{label}_###_{name}.{ending}): files found in directory: "
                     f"{file_counter}")

    for i in range(file_counter):
        try:
            if name in ["EnergyMuon", "EnergyElectron"]:
                data = load_column(path, f"{label}_{i}_{name}.{ending}", 2)
            else:
                data = np.loadtxt(f"{path}{label}_{str(i)}_{name}.{ending}", delimiter=",")
            event_counter += len(data)
        except OSError:
            file_counter -= 1

            data_logger.debug(f"Call to find_number_events - {path}{label}_{i}_{name}.{ending}: file not found")
            continue

    data_logger.info(f"Call to find_number_events ({path}{label}_###_{name}.{ending}): events found in directory: "
                     f"{event_counter}")
    return event_counter


def concat_data_to_arr(arr: np.array, path: str, label: str, name: str, number_of_files: int = -1, ending: str = "csv"
                       ) -> np.array:
    """
    TODO: Check if this function is actually necessary..

    Takes a np array and concatenates data to it, loaded from a specified number of files, according to the
    following pattern:
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number.

    :param arr: np array to be concatenated to, will be returned
    :param number_of_files: How many data files will be loaded; set to -1 if all files should be loaded. Default = -1
    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param name: What data is read in, e.g. 'charge', 'neutron number'
    :param ending: File ending, e.g. 'csv' (without dot). Default = 'csv'
    :return: Returns arr, concatenated with data
    """

    if number_of_files == -1:
        number_of_files = find_number_files(path, label, name, ending)

    i = 0
    while number_of_files > 1:
        i += 1
        try:
            if name in ["EnergyMuon", "EnergyElectron"]:
                data = load_column(path, f"{label}_{i}_{name}.{ending}", 2)
                # TODO
            else:
                data = np.loadtxt(f"{path}{label}_{str(i)}_{name}.{ending}", delimiter=",")
            arr = np.concatenate([arr, data], axis=0)
            number_of_files -= 1
        except OSError:
            data_logger.debug(f"Call to concat_data_to_arr ({path}{label}_{str(i)}_{name}.{ending}): file not found")
            continue

    data_logger.info(f"Call to concat_data_to_arr ({path}{label}_###_{name}.{ending}): successfully loaded "
                     f"the specified files.")
    return arr


def load_data(path: str,
              label: str,
              name: str,
              number_of_files: int = -1,
              ending: str = "csv",
              beam_mrd_coinc: bool = False
              ) -> np.array:
    """
    Similar to concat_data_to_arr, the difference being that an array will be constructed.
    Load data to a np array, loaded from a specified number of files, according to the
    following pattern:
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number.

    :param beam_mrd_coinc: TODO
    :param number_of_files: How many data files will be loaded; set to -1 if all files should be loaded. Default = -1
    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param name: What data is read in, e.g. 'charge', 'neutron number'
    :param ending: File ending, e.g. 'csv' (without dot). Default = 'csv'
    :return: Returns arr, concatenated with data
    """
    arr = []
    if number_of_files == -1:
        number_of_files = find_number_files(path, label, name, ending)

    i = 0
    while number_of_files > 1:
        i += 1
        try:
            if beam_mrd_coinc:
                if name in ["EnergyMuon", "EnergyElectron"]:
                    data = load_column(path, f"R{i}_{label}_{name}.{ending}", 2)
                elif name in ["VisibleEnergy", "Rings"]:
                    data = load_column(path, f"R{i}_{label}_{name}.{ending}", 0)
                elif name in ["MRD"]:
                    data = np.loadtxt(f"{path}R{i}_{label}_{name}.{ending}", delimiter=",")
                else:
                    data = np.loadtxt(f"{path}R{i}_{label}_{name}.{ending}", delimiter=",")
                    np.reshape(data, (-1, 160))
                try:
                    arr = np.concatenate([arr, data], axis=0) if len(arr) > 0 else data
                except ValueError:
                    # Usually due to a .csv file only containing 1 event
                    raise celfa_exceptions.ErrorMismatch
                    pass

            else:
                if name in ["EnergyMuon", "EnergyElectron"]:
                    data = load_column(path, f"{label}_{i}_{name}.{ending}", 2)
                elif name in ["VisibleEnergy", "Rings"]:
                    data = load_column(path, f"{label}_{i}_{name}.{ending}", 0)
                elif name in ["MRD"]:
                    data = np.loadtxt(f"{path}{label}_{i}_{name}.{ending}", delimiter=",")
                else:
                    data = np.loadtxt(f"{path}{label}_{i}_{name}.{ending}", delimiter=",")
                    np.reshape(data, (-1, 160))
                try:
                    arr = np.concatenate([arr, data], axis=0) if len(arr) > 0 else data
                except ValueError:
                    # Usually due to a .csv file only containing 1 event
                    raise celfa_exceptions.ErrorMismatch
                    pass

            number_of_files -= 1
        except OSError:
            data_logger.debug(f"Call to load_data_to_arr ({path}{label}_{i}_{name}.{ending}): file not found")
            continue

    data_logger.info(f"Call to load_data_to_arr ({path}{label}_###_{name}.{ending}): successfully loaded "
                     f"the specified files.")
    return arr


def load_time_data(path: str, label: str, number_of_files: int = -1, ending="csv") -> np.array:
    """
    Similar to concat_time_data, the difference being that an array will be constructed.
    Special util function to load in time data files, specified by the pattern
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number. The optional parameter
    'number_of_files' specifies how many files should be loaded, leave empty for all possible files in the directory.
    Invert order of high time values <-> low time values, by inverting the interval [0,1] to [1,0] (range of data).
    Also check for sum of elements of charge data of an event to be 0, if so, append empty arr.

    Will take in an array, to which the loaded time data will be appended. Compare "concat_data_to_arr".

    :param ending: File ending. Default = "csv"
    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param number_of_files: How many data files will be loaded; set to -1 if all files should be loaded. Default = -1
    :return: time_val, with loaded time data appended to it. (time)
    """
    arr = np.zeros((0, 160))

    if number_of_files == -1:
        number_of_files = find_number_files(path, label, "charge")

    i = 0
    while number_of_files > 1:
        # 'mismatch_flag' monitors which of the 2 files (time, charge) could not be loaded - if only one
        # of the files could not be loaded (mismatch in data files), it will raise a critical error.
        mismatch_flag = 0
        i += 1
        try:
            training_data_list_t = np.loadtxt(f"{path}{label}_{i}_time.{ending}", delimiter=",")
        except OSError:
            data_logger.debug(f"Call to concat_time_data ({path}{label}_{i}_time.{ending}): "
                              f"file not found")
            mismatch_flag += 1
            continue
        try:
            training_data_list_c = np.loadtxt(f"{path}{label}_{i}_charge.{ending}", delimiter=",")
        except OSError:
            mismatch_flag += 1
            data_logger.debug(f"Call to concat_time_data ({path}{label}_{i}_charge.{ending}): file not found")
            continue

        if mismatch_flag > 1:
            data_logger.critical(f"Call to concat_time_data ({path}{label}_{i}): data mismatch error!")
            raise celfa_exceptions.ErrorMismatch

        loaded_from_csv = []
        for element in range(len(training_data_list_t)):
            if sum(training_data_list_c[element]) == 0:
                loaded_from_csv.append(np.zeros(160 * 1))
            else:
                # TODO: this
                temp_flip = [(0 if not k else 1 - k) for k in training_data_list_t[element]]
                loaded_from_csv.append(temp_flip)

        loaded_from_csv = np.array(loaded_from_csv)
        # TODO: check if np.array is a better solution for ragged data shapes
        arr = [*arr, *loaded_from_csv]
        number_of_files -= 1

    data_logger.info(f"Call to concat_time_data ({path}{label}): executed successfully")
    return arr


def concat_time_data(arr: np.array, path: str, label: str, number_of_files: int = -1) -> np.array:
    """
    Special util function to load in time data files, specified by the pattern
    ~/path/label_#####_name.ending, where '#####' is the corresponding file number. The optional parameter
    'number_of_files' specifies how many files should be loaded, leave empty for all possible files in the directory.
    Invert order of high time values <-> low time values, by inverting the interval [0,1] to [1,0] (range of data).
    Also check for sum of elements of charge data of an event to be 0, if so, append empty arr.

    Will take in an array, to which the loaded time data will be appended. Compare "concat_data_to_arr".

    :param path: Directory path (Include forward slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param arr: np array to which the time data will be concatenated.
    :param number_of_files: How many data files will be loaded; set to -1 if all files should be loaded. Default = -1
    :return: time_val, with loaded time data appended to it. (time)
    """
    if number_of_files == -1:
        number_of_files = find_number_files(path, label, "charge")

    i = 0
    while number_of_files > 1:
        # 'mismatch_flag' monitors which of the 2 files (time, charge) could not be loaded - if only one
        # of the files could not be loaded (mismatch in data files), it will raise a critical error.
        mismatch_flag = 0
        i += 1
        try:
            training_data_list_t = np.loadtxt(f"{path}{label}_{str(i)}_time.csv", delimiter=",")
        except OSError:
            data_logger.debug(f"Call to concat_time_data ({path}{label}_{str(i)}_time.csv): file not found")
            mismatch_flag += 1
            continue
        try:
            training_data_list_c = np.loadtxt(f"{path}{label}_{str(i)}_charge.csv", delimiter=",")
        except OSError:
            mismatch_flag += 1
            data_logger.debug(f"Call to concat_time_data ({path}{label}_{str(i)}_charge.csv): file not found")
            continue

        if mismatch_flag > 1:
            data_logger.critical(f"Call to concat_time_data ({path}{label}_{str(i)}): data mismatch error!")
            raise celfa_exceptions.ErrorMismatch

        loaded_from_csv = []
        for element in range(len(training_data_list_t)):
            if sum(training_data_list_c[element]) == 0:
                loaded_from_csv.append(np.zeros(160 * 1))
            else:
                # TODO: this
                temp_flip = [(0 if not k else 1 - k) for k in training_data_list_t[element]]
                loaded_from_csv.append(temp_flip)

        loaded_from_csv = np.array(loaded_from_csv)
        arr = np.concatenate([arr, loaded_from_csv], axis=0)
        number_of_files -= 1

    data_logger.info(f"Call to concat_time_data ({path}{label}): executed successfully")
    return arr


def zip_data(data_sets: list) -> list:
    """
    Zip multiple data sets at once, return the zipped data sets.

    Note: This is a free, single standing function since it relies on the behaviour of zip(). Specifically, the property
    that only as many elements as the length of the shortest list will be zipped together, is used. This allows for
    easy test-implementation, and using the code in an older version of python, or if the behaviour of zip() is changed,
    it is easier to debug, since only this singular function will break.

    :param data_sets: List of data (in the form of arrays)
    :return: np.array of zipped data_sets
    """
    return list(zip(*data_sets))


def construct_data_array(data_names, path: str, label: str, number_of_files: int = -1,
                         ending: str = "csv", load_from_config=False, beam_mrd_coinc: bool = False) -> np.array:
    """
    Return zipped data sets in the order specified by data_names. Use "time" to load time, using the special
    load_time_data function. The order in which the data sets will be zipped is the order of the entries in data_names.

    The file path is given by ~/path/label_#####_name.ending, where '#####' is the corresponding file number.

    Note: Also automatically appends the appropriate data identifier (1, 0) for electron, (0, 1) for muon.
    :param beam_mrd_coinc: TODO
    :param ending: File ending, e.g. 'csv' (without dot). Default = 'csv'
    :param data_names: List of file names which will be loaded, e.g. 'charge', 'neutron_number', 'time'.
    :param path: Directory path (Include slash at the end)
    :param label: Label of the data, e.g. 'electron_beamlike'
    :param number_of_files: How many data files will be loaded; set to -1 if all files should be loaded. Default = -1
    :param load_from_config: TODO: specify from which config file will be loaded, and support loading from config file
    this should essentially replace data_names. This will allow for having multiple configs ready, and training
    a plethora of models in one go with different data configs.
    Default = False.
    :return: After loading, return all data sets zipped together, using zip(). Length of the data sets will be cut to
    that of the shortest data set.
    """
    loaded_data = []
    for name in data_names:
        if name in ["time", "time_abs"]:
            loaded_data.append(load_time_data(path, label, ending=ending))
        else:
            loaded_data.append(load_data(path, label, name, number_of_files=number_of_files,
                                         ending=ending, beam_mrd_coinc=beam_mrd_coinc))

        # Should be unnecessary since zip_data() ensures that each list is the same length, but gives peace of mind
        length = len(loaded_data[0])
        if not all(len(d) == length for d in loaded_data):
            data_logger.critical("Call to construct_data_array: Mismatch in data length!")
            raise celfa_exceptions.ErrorMismatch

    data_logger.info(f"Call to construct_data_array ({path}{label}): executed successfully")
    return zip_data(loaded_data)


def concat_min_array(*darrs: list) -> list:
    """
    Join and return multiple arrays such that as many entries of each array are joined together as the shortest array
    contains. This is useful if an unknown number of events from different categories are loaded into different arrays,
    but each category is wanted to contain the same amount of events after being concatenated.

    >>> concat_min_array([1, 2, 3, 4, 5], [1, 1, 1], [0, 1, 3, 4])
    [[1, 2, 3], [1, 1, 1], [0, 1, 3]]

    :param darrs: Arbitrary number of arrays
    :return: Return multiple arrays such that as many entries of each array are joined together as the shortest array
    contains
    """
    min_l = len(darrs[0])
    for d in darrs:
        if len(d) < min_l:
            min_l = len(d)
    return [d[:min_l] for d in darrs]


def build_category_values(categories: list, num: int) -> list:
    """
    Return a list of category values specified in categories. Appends in order given when calling the function. num
    specifies how many data points are in each category. Assumes every category is the same length!

    :param categories: Specifier for category values; e.g. (1, 0) for electron, (0, 1) for muon -> [(1,0), (0,1)].
    :param num: How many identifiers for each category.
    :return: List of all category values appended to each other; ordering is specified by ordering of the parameter
    categories.
    """
    category_values = []
    for category in categories:
        [category_values.append(category) for __ in range(num)]

    data_logger.info(f"build_category_values: Built category values")
    return category_values


def load_column(path: str, filename: str, col: int, delimiter: str = ",", fill_empty: bool = True,
                fill_value: float = 0.0) -> List:
    """Return the n-th column of a csv-like file."""
    data = []
    with open(f"{path}{filename}", "r") as f:
        for line in f:
            # to avoid empty values in data
            if not (line.split(delimiter))[col] or (line.split(delimiter))[col] == "\n" and fill_empty:
                val = float(fill_value)
            else:
                val = float((line.split(delimiter))[col])
            data.append(val)
        return data


def save_data(path: str, filename: str, data: object, ending: str = "pickle"):
    """Dump data into file using pickle."""
    try:
        pickle_out = open(f"{path}{filename}.{ending}", "wb")
    except OSError:
        data_logger.critical(f"save_data: Could not open file {path}{filename}.{ending}")
        return
    pickle.dump(data, pickle_out, protocol=4)
    pickle_out.close()


def load_and_create_train_val_test_sets(data, category_values, percentages=None):
    """
    Shuffle and create train, validation and test data sets. The relative size can be defined by using the 'percentages'
    parameter. The returned arrays will be of the form (data, category_values) where data is of the form
    (data1, data2, data3) etc.
    """
    if percentages is None:
        percentages = [0.7, 0.15, 0.15]

    arr = list(zip(data, category_values))
    random.shuffle(arr)

    length = len(data)

    train = arr[:int(percentages[0]*length)]
    validation = arr[int(percentages[0]*length):int(percentages[0]*length) + int(percentages[1]*length)]
    test = arr[int(percentages[0]*length) + int(percentages[1]*length):
               int(percentages[0]*length) + int(percentages[1]*length) + int(percentages[2]*length)]
    return train, validation, test


def split_data_cat(data):
    s_data, s_cat = [], []
    for entry in data:
        s_data.append(entry[0])
        s_cat.append(entry[1])
    return s_data, s_cat


def select_data(data, indices):
    s_data = []

    for entry in data:
        temp = []
        for index in indices:
            temp.append(entry[index])
        s_data.append(temp)
    return s_data
