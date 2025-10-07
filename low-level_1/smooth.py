import pandas as pd
import numpy as np

def smooth(x, epoch_time, sample_rate):
    """
    Apply a centered moving average smoothing to the input DataFrame.
    
    This function smooths the data using a sliding window of 0.2 seconds (0.1s before and after each point)
    centered on each time point. It uses pandas' rolling mean for efficient computation.
    
    Parameters:
    -----------
    x : pd.DataFrame
        Input DataFrame containing time-series data (e.g., EEG/ERP epochs) to be smoothed.
        Assumed to have time points as rows or columns; rolling is applied along the time axis.
    epoch_time : float or int (unused in current implementation)
        Placeholder for epoch duration (in seconds); not currently utilized.
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
    
    # Calculate window size (0.2 * sample_rate, because the window spans from i-0.1*sample_rate to i+0.1*sample_rate)
    window_size = int(0.2 * sample_rate) + 1  # Ensure window size is odd to include the center point
    
    # Use rolling method to compute moving average over the sliding window
    x2 = x2.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # For boundaries (first and last 0.1*sample_rate points), rolling automatically fills with smaller windows
    return x2