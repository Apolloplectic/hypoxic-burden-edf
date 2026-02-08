"""
Core PSG Analysis Engine for Hypoxic Burden Calculator
"""

import numpy as np
import pandas as pd
import mne
from scipy.integrate import trapezoid as trapz
from pathlib import Path
import wfdb

from config import YASA_AVAILABLE, CHANNEL_PATTERNS
from utils import detect_channel, calculate_robust_baseline

if YASA_AVAILABLE:
    import yasa


class PSGAnalyzer:
    """
    Main class for analyzing polysomnography (PSG) data
    """
    
    def __init__(self, raw, filepath):
        """
        Initialize PSG analyzer
        
        Parameters:
        -----------
        raw : mne.io.Raw
            MNE raw object containing PSG data
        filepath : str
            Path to the EDF file
        """
        self.raw = raw
        self.filepath = filepath
        
        # Auto-detect channels
        self.spo2_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['spo2'])
        self.flow_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['flow'])
        self.eeg_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['eeg'])
        self.eog_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['eog'])
        self.emg_ch = detect_channel(raw.ch_names, CHANNEL_PATTERNS['emg'])
        
        # MIT annotations
        self.manual_events = []
        self.manual_ahi = None
        self.manual_stages = None
        
        # Processed data
        self.df_spo2 = None
        self.df_flow = None
        self.events_df = None
        self.odi_events = None
        self.stages = None
    
    def check_mit_annotations(self):
        """
        Check for MIT database annotations (.st file)
        
        Returns:
        --------
        bool
            True if MIT annotations found and loaded
        """
        st_path = Path(self.filepath).with_suffix(".st")
        
        if not st_path.exists():
            return False
        
        try:
            # Load annotations
            ann = wfdb.rdann(str(st_path).rsplit(".", 1)[0], "st")
            
            # Extract respiratory events (A = apnea, H = hypopnea)
            resp_idx = [i for i, s in enumerate(ann.symbol) if s and ("A" in s or "H" in s)]
            
            if resp_idx:
                times_sec = ann.sample[resp_idx] / self.raw.info["sfreq"]
                self.manual_events = [{"start": t, "end": t + 10.0} for t in times_sec]
                
                total_sleep_sec = self.raw.times[-1]
                self.manual_ahi = len(self.manual_events) * 3600 / total_sleep_sec
            
            # Extract sleep stages
            stage_symbols = []
            for desc in ann.symbol:
                if desc in ['W', '1', '2', '3', '4']:
                    stage_symbols.append('W' if desc == 'W' else f'N{desc}')
                elif desc == 'R':
                    stage_symbols.append('REM')
                else:
                    stage_symbols.append('Unknown')
            
            if stage_symbols:
                max_epochs = int(self.raw.times[-1] / 30)
                self.manual_stages = stage_symbols[:max_epochs]
            
            return True
        
        except Exception as e:
            print(f"Error loading MIT annotations: {e}")
            return False
    
    def preprocess_spo2(self, artifact_filter='Off'):
        """
        Preprocess SpO₂ signal
        
        Parameters:
        -----------
        artifact_filter : str
            'Off', 'Mild (10%/s)', or 'Strict (5%/s)'
        
        Returns:
        --------
        pd.DataFrame
            Processed SpO₂ data
        """
        # Extract SpO₂ data
        spo2_sig, spo2_times = self.raw[self.spo2_ch]
        
        # Validate data
        if len(spo2_sig.flatten()) != len(spo2_times.flatten()):
            raise ValueError(f"SpO₂ data length mismatch: {len(spo2_sig.flatten())} samples vs {len(spo2_times.flatten())} timepoints")
        
        df = pd.DataFrame({
            "time": spo2_times.flatten(),
            "spo2": spo2_sig.flatten()
        })
        
        # Validate DataFrame
        if df.empty or len(df) == 0:
            raise ValueError("SpO₂ data is empty after extraction")
        
        # Resample to 1 Hz if needed
        if self.raw.info['sfreq'] != 1:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time').resample('1s').mean().interpolate(method='linear').reset_index()
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        
        # Validate after resampling
        if df.empty or len(df) == 0:
            raise ValueError("SpO₂ data is empty after resampling")
        
        # Apply artifact filter
        if artifact_filter != 'Off':
            max_rate = 10 if artifact_filter == "Mild (10%/s)" else 5
            df['rate'] = df['spo2'].diff().abs()
            df['artifact'] = df['rate'] > max_rate
            
            # Mark artifacts as NaN
            df.loc[df['artifact'], 'spo2'] = np.nan
            
            # Interpolate over artifacts
            df['spo2'] = df['spo2'].interpolate(method='linear', limit=5)
        
        self.df_spo2 = df
        return df
    
    def detect_odi_events(self, desat_threshold=3):
        """
        Detect oxygen desaturation events (ODI)
        
        Parameters:
        -----------
        desat_threshold : int
            Desaturation threshold (3 or 4 %)
        
        Returns:
        --------
        pd.DataFrame
            ODI events
        """
        if self.df_spo2 is None:
            raise ValueError("Must run preprocess_spo2() first")
        
        df = self.df_spo2.copy()
        
        # Look for drops ≥ threshold that recover
        df['spo2_next'] = df['spo2'].shift(-10)  # Look 10s ahead
        df['desat'] = (
            (df['spo2'].diff() <= -desat_threshold) &
            (df['spo2_next'] >= df['spo2'] + (desat_threshold - 1))
        )
        
        self.odi_events = df[df['desat']].copy()
        return self.odi_events
    
    def detect_apnea_hypopnea_events(self, desat_threshold=3):
        """
        Detect apnea/hypopnea events from airflow and SpO₂
        
        Parameters:
        -----------
        desat_threshold : int
            Desaturation threshold for event validation
        
        Returns:
        --------
        pd.DataFrame
            Detected events
        """
        if self.df_spo2 is None:
            raise ValueError("Must run preprocess_spo2() first")
        
        if self.flow_ch:
            # Method 1: Use airflow signal
            events = self._detect_from_airflow(desat_threshold)
        else:
            # Method 2: Estimate from SpO₂ desaturations only
            events = self._detect_from_spo2(desat_threshold)
        
        self.events_df = pd.DataFrame(events)
        return self.events_df
    
    def _detect_from_airflow(self, desat_threshold):
        """
        Detect events using airflow signal (more accurate)
        """
        # Resample airflow to 10 Hz
        flow_sig, flow_times = self.raw[self.flow_ch]
        df_flow = pd.DataFrame({
            "time": flow_times.flatten(),
            "flow": flow_sig.flatten()
        })
        
        df_flow['time'] = pd.to_datetime(df_flow['time'], unit='s')
        df_flow = df_flow.set_index('time').resample('0.1s').mean().interpolate(method='linear').reset_index()
        df_flow['time'] = (df_flow['time'] - df_flow['time'].iloc[0]).dt.total_seconds()
        
        self.df_flow = df_flow
        
        # Normalize flow
        flow = df_flow['flow'].values
        t = df_flow['time'].values
        
        peak_flow = np.percentile(np.abs(flow), 95)
        if peak_flow == 0:
            peak_flow = 1
        
        flow_norm = flow / peak_flow
        
        # Calculate rolling baseline (30s window)
        window_size = int(30 / 0.1)  # 300 samples
        baseline = pd.Series(flow_norm).rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).median().values
        
        # Calculate flow reduction from baseline
        reduction = 1 - (flow_norm / (baseline + 1e-6))
        
        # Detect events: ≥30% reduction for ≥10s
        in_event = reduction >= 0.30
        
        # Find event boundaries
        diff = np.diff(np.concatenate(([False], in_event, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Create event list
        events = []
        for s, e in zip(starts, ends):
            duration = t[e-1] - t[s]
            if duration >= 10:  # At least 10 seconds
                events.append({
                    "start": t[s],
                    "end": t[e-1]
                })
        
        # Validate events with SpO₂ desaturation
        valid_events = []
        for ev in events:
            end_t = ev['end']
            
            # Check for desaturation within 30s after event end
            win = self.df_spo2[
                (self.df_spo2['time'] >= end_t) &
                (self.df_spo2['time'] <= end_t + 30)
            ]
            
            if len(win) == 0:
                continue
            
            # Get baseline SpO₂ before event
            pre_win = self.df_spo2[
                (self.df_spo2['time'] >= end_t - 30) &
                (self.df_spo2['time'] < end_t)
            ]
            
            if len(pre_win) == 0:
                continue
            
            baseline_spo2 = pre_win['spo2'].max()
            min_spo2 = win['spo2'].min()
            drop = baseline_spo2 - min_spo2
            
            # Must have ≥3% or ≥4% desaturation
            if drop >= desat_threshold:
                valid_events.append(ev)
        
        return valid_events
    
    def _detect_from_spo2(self, desat_threshold):
        """
        Estimate events from SpO₂ alone (less accurate, used when no airflow)
        """
        # Find significant SpO₂ drops
        drops = self.df_spo2['spo2'].diff() < -desat_threshold
        starts = self.df_spo2[drops].index
        
        # Assume events last ~60s
        ends = (starts + 60).clip(upper=len(self.df_spo2) - 1)
        
        events = []
        for s, e in zip(starts, ends):
            events.append({
                "start": self.df_spo2.loc[s, 'time'],
                "end": self.df_spo2.loc[e, 'time']
            })
        
        return events
    
    def perform_sleep_staging(self, use_mit_st=False):
        """
        Perform sleep staging
        
        Parameters:
        -----------
        use_mit_st : bool
            Use MIT manual annotations if available
        
        Returns:
        --------
        list
            Sleep stages for each 30s epoch
        """
        import streamlit as st
        
        if use_mit_st and self.manual_stages is not None:
            st.info("✅ Using MIT gold standard annotations")
            self.stages = self.manual_stages
            return self.stages
        
        # Check if YASA is available and we have EEG
        if YASA_AVAILABLE:
            if not self.eeg_ch:
                st.warning("⚠️ YASA available but no EEG channel found - using rule-based staging")
            elif self.raw.info['sfreq'] < 100:
                st.warning(f"⚠️ YASA requires ≥100 Hz, got {self.raw.info['sfreq']} Hz - using rule-based staging")
            else:
                st.info(f"🔬 Attempting YASA deep learning staging (sfreq: {self.raw.info['sfreq']} Hz)")
                try:
                    stages = self._yasa_staging()
                    self.stages = stages
                    st.success(f"✅ YASA staging successful: {len(stages)} epochs")
                    return stages
                except Exception as e:
                    st.error(f"❌ YASA failed: {type(e).__name__}: {str(e)}")
                    st.warning("🔄 Falling back to rule-based staging...")
        else:
            st.info("ℹ️ YASA not installed - using rule-based staging")
        
        # Fall back to rule-based staging
        st.info("🔬 Using rule-based spectral staging")
        stages = self._rule_based_staging()
        self.stages = stages
        return stages
    
    def _yasa_staging(self):
        """
        Perform automatic sleep staging using YASA
        """
        try:
            # Check sampling rate
            if self.raw.info['sfreq'] < 100:
                print(f"⚠️ YASA requires ≥100 Hz, got {self.raw.info['sfreq']} Hz")
                raise ValueError(f"Sampling rate too low: {self.raw.info['sfreq']} Hz")
            
            # Try to create SleepStaging object
            print(f"🔬 Attempting YASA staging with EEG: {self.eeg_ch}, EOG: {self.eog_ch}, EMG: {self.emg_ch}")
            
            sls = yasa.SleepStaging(
                self.raw,
                eeg_name=self.eeg_ch,
                eog_name=self.eog_ch,
                emg_name=self.emg_ch
            )
            
            print("✅ YASA SleepStaging object created successfully")
            
            # Try to predict
            hypno = sls.predict()
            print(f"✅ YASA prediction successful: {len(hypno)} epochs")
            
            # Convert YASA stages to standard format
            stages = []
            for stage in hypno:
                if stage == 'W':
                    stages.append('W')
                elif stage == 'N1':
                    stages.append('N1')
                elif stage == 'N2':
                    stages.append('N2')
                elif stage == 'N3':
                    stages.append('N3')
                elif stage == 'R':
                    stages.append('REM')
                else:
                    stages.append('Unknown')
            
            # Crop to recording duration
            max_epochs = int(self.raw.times[-1] / 30)
            stages = stages[:max_epochs]
            
            print(f"✅ YASA staging complete: {len(stages)} epochs converted")
            return stages
            
        except Exception as e:
            print(f"❌ YASA staging failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to trigger fallback
    
    def _rule_based_staging(self):
        """
        Simple rule-based sleep staging (fallback when YASA unavailable)
        """
        if not self.eeg_ch:
            # No EEG = can't stage, return all as "Total"
            n_epochs = int(self.raw.times[-1] / 30)
            return ['Total'] * n_epochs
        
        # Create 30s epochs
        events_mne = mne.make_fixed_length_events(self.raw, id=1, duration=30.0)
        
        try:
            epochs = mne.Epochs(
                self.raw,
                events_mne,
                tmin=0,
                tmax=30.0,
                preload=True,
                picks=[self.eeg_ch, self.eog_ch, self.emg_ch] if self.eog_ch and self.emg_ch else [self.eeg_ch],
                baseline=None,
                verbose=False
            )
        except Exception:
            n_epochs = int(self.raw.times[-1] / 30)
            return ['Total'] * n_epochs
        
        stages = []
        
        for i in range(len(epochs)):
            try:
                epoch = epochs[i]
                
                # Compute power spectral density
                psds, freqs = mne.time_frequency.psd_array_welch(
                    epoch.get_data(picks=self.eeg_ch)[0],
                    sfreq=self.raw.info['sfreq'],
                    fmin=0.5,
                    fmax=30,
                    n_fft=1024,
                    verbose=False
                )
                
                # Calculate band powers
                delta = np.mean(psds[(freqs >= 0.5) & (freqs < 4)])
                theta = np.mean(psds[(freqs >= 4) & (freqs < 8)])
                alpha = np.mean(psds[(freqs >= 8) & (freqs < 12)])
                spindle = np.mean(psds[(freqs >= 12) & (freqs < 15)])
                
                # Simple staging rules
                if alpha > theta * 1.5:
                    stage = 'W'
                elif spindle > theta:
                    stage = 'N2'
                elif delta > theta:
                    stage = 'N3'
                elif theta > alpha:
                    stage = 'N1'
                else:
                    stage = 'REM'
                
                stages.append(stage)
            
            except Exception:
                stages.append('Unknown')
        
        return stages
    
    def calculate_hypoxic_burden(self, pre_event_sec=100, desat_start_sec=60,
                                 desat_end_sec=120, artifact_filter='Off'):
        """
        Calculate obstructive hypoxic burden (event-specific)
        
        Parameters:
        -----------
        pre_event_sec : int
            Pre-event window for baseline calculation (default 100s)
        desat_start_sec : int
            Start of desaturation window before event end (default 60s)
        desat_end_sec : int
            End of desaturation window after event end (default 120s)
        artifact_filter : str
            Artifact filter setting
        
        Returns:
        --------
        dict
            Results including HB, proof events, stage-specific results
        """
        if self.events_df is None or len(self.events_df) == 0:
            # Even with 0 events, return stage breakdown if staging was done
            if self.stages is None:
                self.perform_sleep_staging()
            
            # Calculate stage-specific ODI even without apnea events
            stage_results = {}
            if self.stages:
                stage_counts = pd.Series(self.stages).value_counts()
                stage_time = (stage_counts * 30 / 3600).to_dict()
                
                for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                    hrs = stage_time.get(stage, 0)
                    if hrs == 0:
                        continue
                    
                    # Calculate ODI for this stage
                    stage_indices = [i for i, s in enumerate(self.stages) if s == stage]
                    if stage_indices:
                        stage_start_sec = stage_indices[0] * 30
                        stage_end_sec = stage_start_sec + hrs * 3600
                        
                        odi_in_stage = self.odi_events[
                            (self.odi_events['time'] >= stage_start_sec) &
                            (self.odi_events['time'] < stage_end_sec)
                        ]
                        odi_stage = len(odi_in_stage) / hrs if hrs > 0 else 0
                    else:
                        odi_stage = 0
                    
                    stage_results[stage] = {
                        'hrs': hrs,
                        'AHI': 0.0,
                        'ODI': odi_stage,
                        'HB': 0.0
                    }
            
            return {
                'total_hb': 0.0,
                'proof_events': [],
                'stage_results': stage_results
            }
        
        if self.stages is None:
            self.perform_sleep_staging()
        
        # Organize events by sleep stage
        stage_events = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': [], 'Unknown': [], 'Total': []}
        
        for i, stage in enumerate(self.stages):
            if stage not in stage_events:
                stage = 'Unknown'
            
            start_sec = i * 30
            end_sec = (i + 1) * 30
            
            # Find events in this epoch
            # IMPORTANT: Use 'start' time to attribute events to the sleep stage where they began,
            # not the stage where the arousal occurred (which is often Wake)
            evs = self.events_df[
                (self.events_df['start'] >= start_sec) &
                (self.events_df['start'] < end_sec)
            ]
            
            stage_events[stage].extend(evs.to_dict('records'))
            stage_events['Total'].extend(evs.to_dict('records'))
        
        # Calculate HB for each stage
        stage_results = {}
        proof_events = []
        total_hb_weighted = 0.0
        total_time = 0.0
        
        # Count stage durations
        stage_counts = pd.Series(self.stages).value_counts()
        stage_time = (stage_counts * 30 / 3600).to_dict()  # Convert to hours
        
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            hrs = stage_time.get(stage, 0)
            if hrs == 0:
                continue
            
            evs = stage_events.get(stage, [])
            
            # Calculate HB for this stage
            area_total, stage_proof_events = self._calculate_stage_hb(
                evs, pre_event_sec, desat_start_sec, desat_end_sec, artifact_filter
            )
            
            hb_stage = (area_total / 60) / hrs if hrs > 0 else 0
            
            # Calculate stage-specific ODI
            stage_indices = [i for i, s in enumerate(self.stages) if s == stage]
            if stage_indices:
                stage_start_sec = stage_indices[0] * 30
                stage_end_sec = stage_start_sec + hrs * 3600
                
                odi_in_stage = self.odi_events[
                    (self.odi_events['time'] >= stage_start_sec) &
                    (self.odi_events['time'] < stage_end_sec)
                ]
                odi_stage = len(odi_in_stage) / hrs if hrs > 0 else 0
            else:
                odi_stage = 0
            
            ahi_stage = len(evs) / hrs if hrs > 0 else 0
            
            stage_results[stage] = {
                'hrs': hrs,
                'AHI': ahi_stage,
                'ODI': odi_stage,
                'HB': hb_stage
            }
            
            total_hb_weighted += hb_stage * hrs
            total_time += hrs
            proof_events.extend(stage_proof_events)
        
        # Overall HB (weighted by stage duration)
        total_hb = total_hb_weighted / total_time if total_time > 0 else 0
        
        return {
            'total_hb': total_hb,
            'proof_events': proof_events,
            'stage_results': stage_results
        }
    
    def _calculate_stage_hb(self, events, pre_event_sec, desat_start_sec,
                           desat_end_sec, artifact_filter):
        """
        Calculate HB for events in a specific stage
        """
        area_total = 0.0
        proof_events = []
        
        for ev in events:
            end_t = ev['end']
            
            # Get baseline from pre-event window
            base_df = self.df_spo2[
                (self.df_spo2['time'] >= end_t - pre_event_sec) &
                (self.df_spo2['time'] < end_t)
            ]
            
            if len(base_df) == 0:
                continue
            
            baseline = base_df['spo2'].max()
            
            # Get desaturation window
            win_df = self.df_spo2[
                (self.df_spo2['time'] >= end_t - desat_start_sec) &
                (self.df_spo2['time'] <= end_t + desat_end_sec)
            ].copy()
            
            if len(win_df) < 2:
                continue
            
            # Remove artifacts if filter is enabled
            if artifact_filter != "Off" and 'artifact' in self.df_spo2.columns:
                artifact_indices = self.df_spo2[self.df_spo2['artifact']].index
                win_df = win_df[~win_df.index.isin(artifact_indices)]
            
            # Calculate area below baseline
            depth = np.maximum(baseline - win_df['spo2'].values, 0)
            area = trapz(depth, win_df['time'].values)
            
            area_total += area
            
            # Store for proof plots
            proof_events.append({
                'end_t': end_t,
                'baseline': baseline,
                'win_df': win_df,
                'depth': depth,
                'area': area,
                'hb_contrib': area / 60
            })
        
        return area_total, proof_events
    
    def calculate_global_hypoxic_burden(self, preset_baseline=0.0):
        """
        Calculate global hypoxic burden (whole-study area below baseline)
        
        Parameters:
        -----------
        preset_baseline : float
            Manual baseline SpO₂. If 0, auto-calculate using 95th percentile
        
        Returns:
        --------
        dict
            Global HB and baseline used
        """
        if self.df_spo2 is None:
            raise ValueError("Must run preprocess_spo2() first")
        
        # Determine baseline
        if preset_baseline > 0:
            baseline = preset_baseline
        else:
            # Auto: 95th percentile (filters out desaturations)
            baseline = calculate_robust_baseline(
                self.df_spo2['spo2'].values,
                method='percentile',
                percentile=95
            )
        
        # Calculate area above SpO₂ curve (integral of 100 - SpO₂)
        depth_global = np.maximum(100 - self.df_spo2['spo2'].values, 0)
        area_above_spo2 = trapz(depth_global, self.df_spo2['time'].values)
        
        # Calculate area of rectangle above baseline
        total_sleep_sec = self.df_spo2['time'].max()
        area_above_baseline = (100 - baseline) * total_sleep_sec
        
        # Global desaturation area = difference
        global_desat_area = max(0, area_above_spo2 - area_above_baseline)
        
        # Convert to (%min)/h
        total_hours = total_sleep_sec / 3600
        global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0
        
        return {
            'global_hb': global_hb,
            'baseline': baseline,
            'area': global_desat_area
        }
    
    def calculate_bootstrap_ci(self, n_boot=1000, pre_event_sec=100,
                               desat_start_sec=60, desat_end_sec=120,
                               artifact_filter='Off'):
        """
        Calculate 95% confidence interval using bootstrap
        
        Parameters:
        -----------
        n_boot : int
            Number of bootstrap iterations
        pre_event_sec, desat_start_sec, desat_end_sec : int
            HB calculation parameters
        artifact_filter : str
            Artifact filter setting
        
        Returns:
        --------
        tuple
            (ci_low, ci_high)
        """
        if self.events_df is None or len(self.events_df) == 0:
            return (0.0, 0.0)
        
        hb_values = []
        total_hours = self.raw.times[-1] / 3600
        
        for _ in range(n_boot):
            # Bootstrap sample events (with replacement)
            boot_events = self.events_df.sample(n=len(self.events_df), replace=True)
            
            area_total = 0.0
            
            for _, ev in boot_events.iterrows():
                end_t = ev['end']
                
                # Get baseline
                base_df = self.df_spo2[
                    (self.df_spo2['time'] >= end_t - pre_event_sec) &
                    (self.df_spo2['time'] < end_t)
                ]
                
                if len(base_df) == 0:
                    continue
                
                baseline = base_df['spo2'].max()
                
                # Get desaturation window
                win_df = self.df_spo2[
                    (self.df_spo2['time'] >= end_t - desat_start_sec) &
                    (self.df_spo2['time'] <= end_t + desat_end_sec)
                ]
                
                if len(win_df) < 2:
                    continue
                
                # Remove artifacts
                if artifact_filter != "Off" and 'artifact' in self.df_spo2.columns:
                    artifact_indices = self.df_spo2[self.df_spo2['artifact']].index
                    win_df = win_df[~win_df.index.isin(artifact_indices)]
                
                # Calculate area
                depth = np.maximum(baseline - win_df['spo2'].values, 0)
                area = trapz(depth, win_df['time'].values)
                area_total += area
            
            # HB for this bootstrap sample
            hb_boot = (area_total / 60) / total_hours
            hb_values.append(hb_boot)
        
        # Calculate 95% CI
        ci_low, ci_high = np.percentile(hb_values, [2.5, 97.5])
        
        return (ci_low, ci_high)
    
    def run_full_analysis(self, pre_event_sec=100, desat_start_sec=60,
                         desat_end_sec=120, artifact_filter='Off',
                         desat_threshold=3, use_global_hb=True,
                         preset_baseline=0.0, use_mit_st=False):
        """
        Run complete PSG analysis pipeline
        
        Parameters:
        -----------
        pre_event_sec : int
            Pre-event baseline window
        desat_start_sec : int
            Desaturation window start
        desat_end_sec : int
            Desaturation window end
        artifact_filter : str
            Artifact filter setting
        desat_threshold : int
            Desaturation threshold (3 or 4%)
        use_global_hb : bool
            Calculate global HB
        preset_baseline : float
            Manual baseline for global HB (0 = auto)
        use_mit_st : bool
            Use MIT annotations if available
        
        Returns:
        --------
        dict
            Complete analysis results
        """
        # Step 1: Preprocess SpO₂
        self.preprocess_spo2(artifact_filter)
        
        # Step 2: Detect ODI events
        self.detect_odi_events(desat_threshold)
        
        # Step 3: Detect apnea/hypopnea events
        self.detect_apnea_hypopnea_events(desat_threshold)
        
        # Step 4: Sleep staging
        self.perform_sleep_staging(use_mit_st)
        
        # Step 5: Calculate obstructive HB
        hb_results = self.calculate_hypoxic_burden(
            pre_event_sec, desat_start_sec, desat_end_sec, artifact_filter
        )
        
        # Step 6: Calculate bootstrap CI
        if len(self.events_df) > 0:
            ci_low, ci_high = self.calculate_bootstrap_ci(
                1000, pre_event_sec, desat_start_sec, desat_end_sec, artifact_filter
            )
        else:
            ci_low, ci_high = 0.0, 0.0
        
        # Step 7: Calculate global HB (if requested)
        global_hb_results = None
        if use_global_hb:
            global_hb_results = self.calculate_global_hypoxic_burden(preset_baseline)
        
        # Calculate metrics
        total_hours = self.raw.times[-1] / 3600
        ahi = len(self.events_df) / total_hours if total_hours > 0 else 0
        odi = len(self.odi_events) / total_hours if total_hours > 0 else 0
        
        # Compile results
        results = {
            'duration': total_hours,
            'ahi': ahi,
            'odi': odi,
            'total_hb': hb_results['total_hb'],
            'ci': (ci_low, ci_high),
            'events': hb_results['proof_events'],
            'stage_hb': hb_results['stage_results'],
            'manual_ahi': self.manual_ahi,
            'use_mit_st': use_mit_st and self.manual_stages is not None
        }
        
        # Add global HB if calculated
        if global_hb_results:
            results['global_hb'] = global_hb_results['global_hb']
            results['baseline_used'] = global_hb_results['baseline']
        
        return results
