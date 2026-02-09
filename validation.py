"""
Data Validation Utilities for PSG Analysis
Validates EDF files and signal quality before analysis
"""

import numpy as np
import pandas as pd
import streamlit as st


class PSGValidator:
    """
    Validates PSG data quality and provides helpful warnings
    """
    
    def __init__(self, raw, analyzer):
        """
        Initialize validator
        
        Parameters:
        -----------
        raw : mne.io.Raw
            MNE raw object
        analyzer : PSGAnalyzer
            Initialized PSG analyzer
        """
        self.raw = raw
        self.analyzer = analyzer
        self.warnings = []
        self.errors = []
        self.info = []
    
    def validate_all(self):
        """
        Run all validation checks
        
        Returns:
        --------
        dict
            Validation results with warnings, errors, and info
        """
        self.validate_duration()
        self.validate_spo2_quality()
        self.validate_channels()
        self.validate_sampling_rates()
        
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }
    
    def validate_duration(self):
        """Check if recording duration is adequate"""
        duration_hours = self.raw.times[-1] / 3600
        
        if duration_hours < 2:
            self.errors.append(
                f"Recording too short: {duration_hours:.1f} hours. Minimum 2 hours required."
            )
        elif duration_hours < 4:
            self.warnings.append(
                f"Short recording: {duration_hours:.1f} hours. Sleep staging may be unreliable. "
                "Recommended minimum: 4 hours for accurate stage detection."
            )
        elif duration_hours > 12:
            self.warnings.append(
                f"Long recording: {duration_hours:.1f} hours. This may include wake periods "
                "before/after sleep. Results will include entire recording period."
            )
        else:
            self.info.append(
                f"✅ Good recording duration: {duration_hours:.1f} hours"
            )
    
    def validate_spo2_quality(self):
        """Validate SpO₂ signal quality"""
        if not self.analyzer.spo2_ch:
            return
        
        # Extract SpO₂ data
        spo2_data, _ = self.raw[self.analyzer.spo2_ch]
        spo2_values = spo2_data.flatten()
        
        # Check for all zeros
        if np.all(spo2_values == 0):
            self.errors.append(
                "SpO₂ channel contains all zeros. Signal not recorded properly."
            )
            return
        
        # Check for all 100s
        if np.all(spo2_values == 100):
            self.errors.append(
                "SpO₂ channel contains all 100% values. Sensor may not be connected."
            )
            return
        
        # Check for unrealistic values
        valid_range = (spo2_values >= 50) & (spo2_values <= 100)
        pct_valid = np.sum(valid_range) / len(spo2_values) * 100
        
        if pct_valid < 50:
            self.errors.append(
                f"SpO₂ data quality poor: only {pct_valid:.1f}% of values in valid range (50-100%). "
                "Check sensor connection and file integrity."
            )
        elif pct_valid < 90:
            self.warnings.append(
                f"SpO₂ data quality marginal: {pct_valid:.1f}% of values in valid range. "
                "Results may be less accurate. Consider artifact filtering."
            )
        else:
            self.info.append(
                f"✅ SpO₂ data quality good: {pct_valid:.1f}% valid values"
            )
        
        # Check for flat signal (no variation)
        spo2_std = np.std(spo2_values[valid_range])
        if spo2_std < 0.5:
            self.warnings.append(
                f"SpO₂ signal very flat (std dev: {spo2_std:.2f}%). "
                "May indicate sensor issue or patient on supplemental oxygen."
            )
        
        # Check for mean SpO₂
        mean_spo2 = np.mean(spo2_values[valid_range])
        if mean_spo2 < 88:
            self.warnings.append(
                f"Low mean SpO₂: {mean_spo2:.1f}%. Patient may have chronic hypoxemia "
                "or severe OSA. Global HB will be very high."
            )
        elif mean_spo2 > 98:
            self.info.append(
                f"High mean SpO₂: {mean_spo2:.1f}%. Patient may be on supplemental oxygen "
                "or have excellent oxygenation."
            )
    
    def validate_channels(self):
        """Validate presence and quality of required channels"""
        # SpO₂ (required)
        if not self.analyzer.spo2_ch:
            self.errors.append(
                "No SpO₂ channel detected. Required for analysis."
            )
        
        # Airflow (recommended)
        if not self.analyzer.flow_ch:
            self.warnings.append(
                "No airflow channel detected. Using SpO₂-based event detection only. "
                "Accuracy may be reduced."
            )
        else:
            self.info.append(
                f"✅ Airflow channel found: {self.analyzer.flow_ch}"
            )
        
        # EEG (for staging)
        if not self.analyzer.eeg_ch:
            self.warnings.append(
                "No EEG channel detected. Sleep staging will not be available. "
                "Stage-specific metrics cannot be calculated."
            )
        else:
            self.info.append(
                f"✅ EEG channel found: {self.analyzer.eeg_ch}"
            )
        
        # EOG/EMG (for better staging)
        if self.analyzer.eeg_ch and not self.analyzer.eog_ch:
            self.info.append(
                "No EOG channel detected. REM detection accuracy may be reduced."
            )
        
        if self.analyzer.eeg_ch and not self.analyzer.emg_ch:
            self.info.append(
                "No EMG channel detected. REM vs Wake distinction may be less accurate."
            )
    
    def validate_sampling_rates(self):
        """Check if sampling rates are adequate"""
        sfreq = self.raw.info['sfreq']
        
        # General check
        if sfreq < 10:
            self.warnings.append(
                f"Low sampling rate: {sfreq} Hz. Some features may not work properly."
            )
        
        # EEG-specific check for YASA
        if self.analyzer.eeg_ch:
            if sfreq < 100:
                self.warnings.append(
                    f"EEG sampling rate ({sfreq} Hz) below YASA requirement (100 Hz). "
                    "Will use rule-based sleep staging instead."
                )
            else:
                self.info.append(
                    f"✅ EEG sampling rate adequate for YASA: {sfreq} Hz"
                )
    
    def display_results(self, show_info=False):
        """
        Display validation results in Streamlit UI
        
        Parameters:
        -----------
        show_info : bool
            Whether to show info messages (default: False)
        """
        # Errors (blocking)
        if self.errors:
            st.error("❌ **Critical Issues Found:**")
            for error in self.errors:
                st.error(f"• {error}")
        
        # Warnings (non-blocking)
        if self.warnings:
            with st.expander("⚠️ Warnings & Recommendations", expanded=True):
                for warning in self.warnings:
                    st.warning(f"• {warning}")
        
        # Info (optional)
        if show_info and self.info:
            with st.expander("ℹ️ Data Quality Info", expanded=False):
                for info in self.info:
                    st.info(f"• {info}")


def check_zero_events(results):
    """
    Check if analysis resulted in zero events and provide helpful message
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary
    
    Returns:
    --------
    bool
        True if potential issue detected
    """
    if results['ahi'] == 0 and results['odi'] == 0:
        st.warning(
            "⚠️ **Zero Events Detected**\n\n"
            "This could indicate:\n"
            "- Patient has no sleep apnea (good!)\n"
            "- Detection sensitivity too low (try 'Aggressive' preset)\n"
            "- Airflow channel missing/poor quality\n"
            "- File is corrupted or incomplete\n\n"
            "**Recommendation:** Verify SpO₂ shows desaturations visually. "
            "If desaturations exist but weren't detected, try the Aggressive preset."
        )
        return True
    return False


def check_unrealistic_hb(results):
    """
    Check if hypoxic burden values are unrealistically high/low
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary
    
    Returns:
    --------
    bool
        True if potential issue detected
    """
    total_hb = results.get('total_hb', 0)
    
    # Very high HB (>200)
    if total_hb > 200:
        st.warning(
            f"⚠️ **Unusually High Hypoxic Burden: {total_hb:.1f}**\n\n"
            "This could indicate:\n"
            "- Very severe OSA with profound desaturations\n"
            "- Chronic hypoxemia (low baseline SpO₂)\n"
            "- Sensor artifact/malfunction\n\n"
            f"**Baseline SpO₂:** {results.get('baseline_used', 'N/A'):.1f}%\n"
            "**Recommendation:** Verify SpO₂ signal quality and baseline calculation."
        )
        return True
    
    # Global HB much higher than obstructive HB
    global_hb = results.get('global_hb')
    if global_hb and total_hb > 0:
        ratio = global_hb / total_hb
        if ratio > 5:
            st.info(
                f"ℹ️ **Global HB ({global_hb:.1f}) much higher than Obstructive HB ({total_hb:.1f})**\n\n"
                "This suggests oxygen desaturations occur throughout sleep, "
                "not just during apnea events. Possible causes:\n"
                "- Hypoventilation\n"
                "- Lung disease (COPD, restrictive disease)\n"
                "- Positional desaturations\n"
                "- REM-related hypoventilation"
            )
            return True
    
    return False
