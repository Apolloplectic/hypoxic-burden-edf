"""
Configuration and constants for Hypoxic Burden Calculator
"""

# Try to import YASA
try:
    import yasa
    YASA_AVAILABLE = True
except ImportError:
    YASA_AVAILABLE = False
    yasa = None

# Default analysis parameters (Azarbarzin et al. 2019)
DEFAULT_PARAMS = {
    'pre_event_sec': 100,  # Pre-event baseline window
    'desat_start_sec': 0,   # Offset from event START (0 for Azarbarzin method)
    'desat_end_sec': 90,    # Seconds after event END (90 for Azarbarzin method)
    'desat_threshold': 3,    # AASM standard (3% or 4%)
    'artifact_filter': 'Off',  # 'Off', 'Mild (10%/s)', 'Strict (5%/s)'
    'use_global_hb': True,    # Calculate global hypoxic burden
    'preset_baseline': 0.0,   # 0 = auto (95th percentile), >0 = manual
}

# Channel name patterns for auto-detection
CHANNEL_PATTERNS = {
    'spo2': ['SPO2', 'SAO2', 'SaO2', 'SpO2'],
    'flow': [
        'AIRFLOW', 'FLOW', 'PFlow', 'PFLW', 'NASAL',
        # Abbreviated nasal airflow (common in European PSG systems)
        'NAF2P', 'NAF1', 'NAF',
        # Thermistor / thermocouple
        'THERM',
    ],
    'eeg': [
        # Central channels preferred for sleep staging (YASA needs C3/C4/CZ)
        # --- Central montages first ---
        'C3M2', 'C3-M2', 'C4M1', 'C4-M1',
        'C3-A2', 'C4-A1', 'C3-A1', 'C4-A2',
        'CZ-A1', 'CZ-A2', 'CZ2-A1',
        # --- Then frontal ---
        'F3M2', 'F3-M2', 'F4M1', 'F4-M1',
        'F3-A2', 'F4-A1', 'F3-A1', 'F4-A2',
        'FP1-A2', 'FP2-A1', 'FP1-A1', 'FP2-A2',
        # --- Then occipital ---
        'O1M2', 'O1-M2', 'O2M1', 'O2-M1',
        'O1-A2', 'O2-A1', 'O1-A1', 'O2-A2',
        # Generic EEG labels
        'EEG',
        # Bare electrode names (short — must come AFTER more specific patterns)
        'C3', 'C4', 'CZ', 'F3', 'F4',
    ],
    'eog': [
        'E1M2', 'E1-M2', 'E2M2', 'E2-M2',
        'REOGM2', 'LEOGM2',
        'EOG', 'ROC', 'LOC',
        # E1/E2 without reference (common short form)
        'E1-', 'E2-',
    ],
    'emg': ['CEMG', 'EMG', 'CHIN', 'Chin'],
}

# Risk stratification thresholds (based on Azarbarzin et al.)
RISK_THRESHOLDS = {
    'low': 0,
    'moderate': 20,
    'high': 53,
    'very_high': 88,
}

def get_risk_level(hb):
    """Get risk level string from HB value using centralized thresholds."""
    if hb < RISK_THRESHOLDS['moderate']:
        return "Low"
    elif hb < RISK_THRESHOLDS['high']:
        return "Moderate"
    elif hb < RISK_THRESHOLDS['very_high']:
        return "High"
    else:
        return "Very High"


# File size limits
FILE_SIZE_LIMITS = {
    'online_single': 200,  # MB
    'online_batch_total': 1024,  # MB (1 GB)
    'online_batch_count': 5,
    'local': 4096,  # MB (4 GB) when running locally
}
