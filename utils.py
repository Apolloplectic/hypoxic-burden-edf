"""
Utility functions for Hypoxic Burden Calculator
"""

import streamlit as st
import mne
import os
import tempfile


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    if 'use_mit_st' not in st.session_state:
        st.session_state.use_mit_st = False
    
    if 'analysis_params' not in st.session_state:
        from config import DEFAULT_PARAMS
        st.session_state.analysis_params = DEFAULT_PARAMS.copy()


def load_edf_file(uploaded_file, filename="temp_upload.edf"):
    """
    Load an EDF file from Streamlit uploader

    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
    filename : str
        Temporary filename to use (ignored, kept for API compatibility)

    Returns:
    --------
    raw : mne.io.Raw or None
        Loaded EDF data
    temp_path : str
        Path to temporary file
    """
    try:
        # Save uploaded file to a unique temp location to avoid race conditions
        suffix = os.path.splitext(uploaded_file.name)[1] if hasattr(uploaded_file, 'name') else '.edf'
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="hb_upload_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception:
            os.close(fd)
            raise

        # Load with MNE — preload=False to avoid allocating all channels upfront.
        # Large files (23 channels × high sfreq × hours) can exceed available RAM
        # when preloaded as float64. We load lazily, then selectively preload
        # only the channels the analyzer needs.
        raw = mne.io.read_raw_edf(temp_path, preload=False, verbose=False)

        # Determine which channels we actually need (SpO₂ required; flow, EEG,
        # EOG, EMG optional). Load only those to keep memory footprint small.
        from config import CHANNEL_PATTERNS
        needed = []
        for _category, patterns in CHANNEL_PATTERNS.items():
            ch = detect_channel(raw.ch_names, patterns)
            if ch and ch not in needed:
                needed.append(ch)

        if needed:
            try:
                raw.pick(needed)
            except Exception:
                pass  # If pick fails, fall through to full preload

        raw.load_data(verbose=False)

        return raw, temp_path

    except Exception as e:
        st.error(f"Error loading EDF file: {str(e)}")
        return None, None


def detect_channel(channel_list, patterns):
    """
    Detect a channel from a list based on name patterns.

    Iterates patterns first (in priority order), then channels.
    This ensures higher-priority patterns (e.g., central EEG channels
    for sleep staging) are matched before lower-priority ones, regardless
    of channel ordering in the EDF file.

    Parameters:
    -----------
    channel_list : list
        List of channel names from the EDF file
    patterns : list
        List of string patterns to match, in priority order

    Returns:
    --------
    str or None
        Matched channel name or None
    """
    for pattern in patterns:
        for ch in channel_list:
            if pattern.upper() in ch.upper():
                return ch
    return None


def calculate_robust_baseline(spo2_values, method='percentile', percentile=95):
    """
    Calculate robust baseline SpO₂ that filters out desaturations
    
    Parameters:
    -----------
    spo2_values : array-like
        SpO₂ values
    method : str
        'percentile' or 'mean_upper'
    percentile : float
        Percentile to use (default 95th)
    
    Returns:
    --------
    float
        Baseline SpO₂ value
    """
    import numpy as np
    
    if method == 'percentile':
        # Use Nth percentile (default 95th)
        # This excludes the lowest 5% which are likely desaturations
        baseline = np.percentile(spo2_values[~np.isnan(spo2_values)], percentile)
    
    elif method == 'mean_upper':
        # Calculate mean of upper 50% of values
        clean_values = spo2_values[~np.isnan(spo2_values)]
        median = np.median(clean_values)
        upper_values = clean_values[clean_values >= median]
        baseline = np.mean(upper_values)
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    return baseline


def format_duration(seconds):
    """
    Format duration in seconds to human-readable string
    
    Parameters:
    -----------
    seconds : float
        Duration in seconds
    
    Returns:
    --------
    str
        Formatted duration (e.g., "7h 32m")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def validate_edf_file(raw):
    """
    Validate that EDF file has required channels
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE raw object
    
    Returns:
    --------
    bool, str
        (is_valid, error_message)
    """
    from config import CHANNEL_PATTERNS
    
    # Check for SpO₂ (required)
    spo2_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['spo2'])
    if not spo2_ch:
        return False, ["SpO₂ channel not found. This channel is required for analysis."]

    # Warnings for missing optional channels
    warnings = []

    flow_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['flow'])
    if not flow_ch:
        warnings.append("Airflow channel not found - AHI will be estimated from SpO₂ only")

    eeg_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['eeg'])
    if not eeg_ch:
        warnings.append("EEG channel not found - sleep staging will be limited")

    return True, warnings


def cleanup_temp_files(pattern="temp_*.edf"):
    """
    Clean up temporary EDF files
    
    Parameters:
    -----------
    pattern : str
        File pattern to match (glob)
    """
    import glob
    
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
        except Exception:
            pass


# Logging utility (for debugging)
def log_analysis_step(step_name, details=None):
    """
    Log analysis step for debugging
    
    Parameters:
    -----------
    step_name : str
        Name of the analysis step
    details : dict, optional
        Additional details to log
    """
    import logging
    
    logger = logging.getLogger('HB_Calculator')
    
    if details:
        logger.info(f"{step_name}: {details}")
    else:
        logger.info(step_name)
