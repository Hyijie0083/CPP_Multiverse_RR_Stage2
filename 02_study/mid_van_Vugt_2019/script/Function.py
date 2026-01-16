import numpy as np
import pandas as pd
import os
import scipy
import mne 
import re

PREPROCESSED_DATA_ROOT_MEM = '../data/preprocessData/Memory'
PREPROCESSED_DATA_ROOT_PERC = '../data/preprocessData/Perception'

RESULTS_MEM = '../results/Memory'
RESULTS_PERC = '../results/Perception'

BIDS_TASK_LABEL_MEM = '_task-memoryBasedDecisionMaking'
BIDS_TASK_LABEL_PERC = '_task-perceptionBasedDecisionMaking'
BIDS_BEH_SUFFIX = '_beh'
BIDS_EEG_SUFFIX = '_eeg'

SAMPLING_RATE = 500
CHANNEL_INDICES = [26, 27, 28]

T_ERP_START, T_ERP_END = -1.0, 0.1
T_BASELINE_START, T_BASELINE_END = -1.0, -0.8
T_AMS_SLPS_START, T_AMS_SLPS_END = -0.18, -0.08
T_PAMS_START, T_PAMS_END = -0.05, 0.05

# ==============================
# Helper Functions
# =============================

def load_behavior_data(sub_id, task_mode):
    if task_mode == 'mem':
        path = PREPROCESSED_DATA_ROOT_MEM
        BIDS_TASK_LABEL = BIDS_TASK_LABEL_MEM
    else:
        path = PREPROCESSED_DATA_ROOT_PERC
        BIDS_TASK_LABEL = BIDS_TASK_LABEL_PERC
    
    # Set path for behavior
    path_behavior = os.path.join(path,
                                 'sub-'+sub_id,
                                 'beh',
                                 'sub-'+sub_id+BIDS_TASK_LABEL+BIDS_BEH_SUFFIX+'.tsv')
    assert os.path.exists(path_behavior), f"{sub_id} : behavior file not found: {path_behavior}"
    
    # Load behavior data
    data_behavior = pd.read_csv(path_behavior, sep='\t', header=0)
    data_behavior['subj_idx'] = sub_id

    return data_behavior

def calculate_condition(data_behavior, task_mode):
    # Set mapping rules
    if task_mode == 'mem':
        mapping = {1: -3, 2: 0, 3: 3}
    elif task_mode == 'perc':
        mapping = {1: -2, 2: 0, 3: 2}

    # Extract number of each item
    def extract_numbers(str):
        if pd.isna(str):
            return []
        return [float(x) for x in re.findall(r'\d+', str)]

    data_behavior['item_1_number'] = data_behavior['item_1'].apply(extract_numbers)
    data_behavior['item_2_number'] = data_behavior['item_2'].apply(extract_numbers)
    data_behavior['item_cue_number'] = data_behavior['item_cue'].apply(extract_numbers)

    # Employ mapping for each item
    def map_to_pca(num_list):
        if len(num_list) == 0:
            return []
        return [mapping[int(x)] for x in num_list]
    
    data_behavior['item_1_number'] = data_behavior['item_1_number'].apply(map_to_pca)
    data_behavior['item_2_number'] = data_behavior['item_2_number'].apply(map_to_pca)
    data_behavior['item_cue_number'] = data_behavior['item_cue_number'].apply(map_to_pca)
    
    # Different conditions have different algorithm to get values
    if task_mode == 'mem':
        # Define a function to acquire the type of trials (match or non-matc)
        def is_match_mem(row):
            cue = np.array(row['item_cue_number'])
            item_1 = np.array(row['item_1_number'])
            item_2 = np.array(row['item_2_number'])
            return np.array_equal(cue, item_1) or np.array_equal(cue, item_2)
        
        data_behavior['trial_type'] = data_behavior.apply(is_match_mem, axis=1).map({True: 'match', False: 'non-match'})

        # Compute summed_similarity for ALL trials
        summed_similarities = []
        for _, row in data_behavior.iterrows():
            cue = np.array(row['item_cue_number'])
            item_1 = np.array(row['item_1_number'])
            item_2 = np.array(row['item_2_number'])
            distance_1 = np.linalg.norm(cue - item_1)
            distance_2 = np.linalg.norm(cue - item_2)
            max_distance = np.sqrt(108.0)  # sqrt(3 * 6^2)
            sim1 = 1 - distance_1 / max_distance
            sim2 = 1 - distance_2 / max_distance
            summed_similarities.append(sim1 + sim2)
        data_behavior['summed_similarity'] = summed_similarities

        # Binning: only on non-match trials
        data_behavior['similarity_bin'] = pd.NA
        non_match_mask = data_behavior['trial_type'] == 'non-match'
        vals = data_behavior.loc[non_match_mask, 'summed_similarity'] 
        bins = pd.qcut(vals, q=4, labels=[1, 2, 3, 4], duplicates='drop')
        data_behavior.loc[non_match_mask, 'similarity_bin'] = bins.astype('Int64')

    elif task_mode == 'perc':
        # Define a function to acquire the type of trials (match or non-matc)
        def is_match_perc(row):
            item_1 = np.array(row['item_1_number'])
            item_2 = np.array(row['item_2_number'])
            return np.array_equal(item_1, item_2)
        
        data_behavior['trial_type'] = data_behavior.apply(is_match_perc, axis=1).map({True: 'match', False: 'non-match'})
        
        # Compute summed_similarity for ALL trials
        similarities = []
        for _, row in data_behavior.iterrows():
            item_1 = np.array(row['item_1_number'])
            item_2 = np.array(row['item_2_number'])
            distance = np.linalg.norm(item_1 - item_2)
            max_distance = np.sqrt(48.0)  # sqrt(3 * 4^2)
            sim = 1 - distance / max_distance
            similarities.append(sim)
        data_behavior['similarity'] = similarities

        # Binning: only on non-match trials
        data_behavior['similarity_bin'] = pd.NA
        non_match_mask = data_behavior['trial_type'] == 'non-match'
        vals = data_behavior.loc[non_match_mask, 'similarity'] 
        bins = pd.qcut(vals, q=4, labels=[1, 2, 3, 4], duplicates='drop')
        data_behavior.loc[non_match_mask, 'similarity_bin'] = bins.astype('Int64')
    
    return data_behavior

def load_eeg_data(sub_id, task_mode):
    if task_mode == 'mem':
        path = PREPROCESSED_DATA_ROOT_MEM
        BIDS_TASK_LABEL = BIDS_TASK_LABEL_MEM
    else:
        path = PREPROCESSED_DATA_ROOT_PERC
        BIDS_TASK_LABEL = BIDS_TASK_LABEL_PERC
    path_eeg = os.path.join(path,
                            'sub-'+sub_id,
                            'eeg',
                            'sub-'+sub_id+BIDS_TASK_LABEL+BIDS_EEG_SUFFIX+'.vhdr')
    assert os.path.exists(path_eeg), f"{sub_id} : eeg file not found: {path_eeg}"

    # Load eeg data
    eeg_bids = mne.io.read_raw_brainvision(path_eeg, preload=False)
    events, event_id = mne.events_from_annotations(eeg_bids)

    return eeg_bids, events, event_id

def remove_rt_outlier(data_behavior, events, event_id, events_id_response, RT_OUTLIER_STD_MULTIPLIER=3):

    # Calculate the boundary of rt
    rt_mean = np.nanmean(data_behavior['rt'])
    rt_std = np.nanstd(data_behavior['rt'])
    rt_lower = rt_mean - RT_OUTLIER_STD_MULTIPLIER * rt_std
    rt_upper = rt_mean + RT_OUTLIER_STD_MULTIPLIER * rt_std

    mask_data_behavior_clean = (
        (data_behavior['rt'] >= rt_lower) &
        (data_behavior['rt'] <= rt_upper)
    )

    index_data_behavior_clean = np.where(mask_data_behavior_clean)[0]

    # Remove outlier trials
    data_behavior_clean = data_behavior.iloc[index_data_behavior_clean].reset_index(drop=True)

    # Remove outlier events
    event_id_key_response = {events_id_response: event_id[events_id_response]}
    events_response = events[events[:, 2] == event_id_key_response[events_id_response]]
    events_clean = events_response[index_data_behavior_clean]

    n_data_behavior_clean = len(data_behavior_clean)
    n_events_clean = len(events_clean)
    # Confirm that the number of response events matches the number of behavioral trials
    assert len(events_clean) == len(data_behavior_clean), f"Trial count mismatch! EEG: {n_events_clean}, Behavior: {n_data_behavior_clean}"

    return data_behavior_clean, events_clean, event_id_key_response


def epoch_save_CPP(sub_id, eeg_bids, events_clean, event_id_key_response, task_mode):
    # Set path to save CPP and joint modeling
    if task_mode == 'mem':
        path_results = RESULTS_MEM
    else:
        path_results = RESULTS_PERC
    path_results_cpp = os.path.join(path_results, 'cpp')
    os.makedirs(path_results_cpp, exist_ok=True)

    # Epoch full eeg data
    data_eeg_epochs = mne.Epochs(
        raw = eeg_bids, 
        events = events_clean, 
        event_id = event_id_key_response,
        tmin = T_ERP_START, tmax = T_ERP_END,
        baseline = (T_BASELINE_START, T_BASELINE_END), 
        preload=True
    )

    # Select 3 certain channels for erp and averge channels and trials for CPP
    data_erp = data_eeg_epochs.get_data(picks=CHANNEL_INDICES)
    data_cpp = np.nanmean(data_erp, axis=(0, 1))

    # save CPP data into .mat files
    scipy.io.savemat(os.path.join(path_results_cpp, f'cpp_{sub_id}.mat'), {'CPP': data_cpp})

    return data_cpp

def smooth(x, sample_rate):
    """
    Apply a centered moving average smoothing to the input DataFrame.
    
    This function smooths the data using a sliding window of 0.2 seconds (0.1s before and after each point)
    centered on each time point. It uses pandas' rolling mean for efficient computation.
    
    Parameters:
    -----------
    x : pd.DataFrame
        Input DataFrame containing time-series data (e.g., EEG/ERP epochs) to be smoothed.
        Assumed to have time points as rows or columns; rolling is applied along the time axis.
    sample_rate : int or float
        Sampling rate of the data (Hz), used to compute the window size in samples.
    
    Returns:
    --------
    pd.DataFrame
        Smoothed copy of the input DataFrame.
    
    Notes:
    ------
    - Window size is calculated as 0.2 * sample_rate + 1 to ensure an odd number of points (centered).
    - Boundaries are handled with min_periods=1, using available data for partial windows.
    - This is a low-pass filter approximation suitable for reducing noise in neural time-series data.
    """
    # Create a copy to avoid modifying the original DataFrame
    x2 = x.copy()
    
    # Calculate window size as 51 according to stage 1
    window_size = int(51) 
    
    # Use rolling method to compute moving average over the sliding window
    x2 = x2.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # For boundaries (first and last 0.1*sample_rate points), rolling automatically fills with smaller windows
    return x2