"""
Core PSG Analysis Engine for Hypoxic Burden Calculator
"""

import logging
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

logger = logging.getLogger('HB_Calculator')


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
        self.df_spo2_raw = None  # Pre-artifact-filter copy for visualization
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

            # Separate respiratory events from sleep stages
            # Respiratory events contain 'A' (apnea) or 'H' (hypopnea)
            # Sleep stages are single-char: W, 1, 2, 3, 4, R
            stage_chars = {'W', '1', '2', '3', '4', 'R'}

            resp_idx = [i for i, s in enumerate(ann.symbol)
                        if s and ("A" in s or "H" in s)]

            if resp_idx:
                times_sec = ann.sample[resp_idx] / self.raw.info["sfreq"]
                self.manual_events = [{"start": t, "end": t + 10.0} for t in times_sec]

                total_sleep_sec = self.raw.times[-1]
                self.manual_ahi = len(self.manual_events) * 3600 / total_sleep_sec

            # Extract sleep stages (only symbols that are exactly stage characters)
            stage_indices = [i for i, s in enumerate(ann.symbol)
                            if s in stage_chars]
            stage_symbols = []
            for i in stage_indices:
                desc = ann.symbol[i]
                if desc == 'W':
                    stage_symbols.append('W')
                elif desc == 'R':
                    stage_symbols.append('REM')
                elif desc in ('1', '2', '3', '4'):
                    stage_symbols.append(f'N{desc}')

            if stage_symbols:
                max_epochs = int(self.raw.times[-1] / 30)
                self.manual_stages = stage_symbols[:max_epochs]

            return True
        
        except Exception as e:
            logger.warning("Error loading MIT annotations: %s", e)
            return False
    
    def preprocess_spo2(self, artifact_filter='Off'):
        """
        Preprocess SpO₂ signal with physiological validation.

        Steps:
        1. Extract raw SpO₂ data
        2. Clamp to physiological range [0, 100] and remove non-physiological values
        3. Resample to 1 Hz (if higher sample rate)
        4. Apply artifact filter (rate-of-change based) if enabled
        5. Store raw (pre-artifact-filter) copy for visualization

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

        # --- Physiological validation (ALWAYS applied) ---
        # Mark non-physiological values as NaN (SpO₂ must be 0-100%)
        n_before = len(df)
        non_physio = (df['spo2'] < 0) | (df['spo2'] > 100)
        n_non_physio = non_physio.sum()
        if n_non_physio > 0:
            logger.info("Removed %d non-physiological SpO₂ values (outside 0-100%%)", n_non_physio)
            df.loc[non_physio, 'spo2'] = np.nan

        # Also mark very low values (< 50%) as likely artifact/disconnection
        very_low = df['spo2'] < 50
        n_very_low = very_low.sum()
        if n_very_low > 0:
            logger.info("Marked %d SpO₂ values below 50%% as artifact", n_very_low)
            df.loc[very_low, 'spo2'] = np.nan

        # Interpolate over NaN gaps (up to 10 samples before resampling)
        df['spo2'] = df['spo2'].interpolate(method='linear', limit=10)

        # Drop any remaining NaN at edges
        df = df.dropna(subset=['spo2']).reset_index(drop=True)

        # Resample to 1 Hz if needed
        if self.raw.info['sfreq'] != 1:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time').resample('1s').mean().interpolate(method='linear').reset_index()
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        # Validate after resampling
        if df.empty or len(df) == 0:
            raise ValueError("SpO₂ data is empty after resampling")

        # Clamp again after resampling (interpolation edge effects)
        df['spo2'] = df['spo2'].clip(0, 100)

        # Store raw (pre-artifact-filter) copy for visualization
        df_raw = df.copy()

        # --- Apply artifact filter (rate-of-change based) ---
        if artifact_filter != 'Off':
            max_rate = 10 if artifact_filter == "Mild (10%/s)" else 5
            df['rate'] = df['spo2'].diff().abs()
            df['artifact'] = df['rate'] > max_rate

            n_artifacts = df['artifact'].sum()
            if n_artifacts > 0:
                logger.info("Artifact filter (%s): marked %d samples as artifacts", artifact_filter, n_artifacts)

            # Mark artifacts as NaN
            df.loc[df['artifact'], 'spo2'] = np.nan

            # Interpolate over artifacts
            df['spo2'] = df['spo2'].interpolate(method='linear', limit=5)

        # Store both versions
        self.df_spo2 = df
        self.df_spo2_raw = df_raw  # For visualization (before artifact filtering)
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

        spo2 = self.df_spo2['spo2'].values
        times = self.df_spo2['time'].values
        n = len(spo2)

        # Rolling baseline: max SpO2 in preceding 120s window
        window_sec = 120
        events = []
        i = 0
        while i < n:
            # Baseline: max SpO2 in the 120s before this point
            base_start = max(0, i - window_sec)
            baseline = np.nanmax(spo2[base_start:i]) if i > 0 else spo2[0]

            # Check if current value is ≥ threshold below baseline
            if baseline - spo2[i] >= desat_threshold:
                # Found a desaturation — find the nadir
                nadir_idx = i
                while nadir_idx + 1 < n and spo2[nadir_idx + 1] <= spo2[nadir_idx]:
                    nadir_idx += 1

                # Check for recovery (rise ≥ threshold-1 from nadir within 120s)
                recovery_end = min(n, nadir_idx + window_sec)
                recovered = False
                for j in range(nadir_idx + 1, recovery_end):
                    if spo2[j] - spo2[nadir_idx] >= (desat_threshold - 1):
                        recovered = True
                        break

                if recovered:
                    events.append({'time': times[i], 'spo2': spo2[nadir_idx]})

                # Skip past this event to avoid double-counting
                i = nadir_idx + 1
            else:
                i += 1

        self.odi_events = pd.DataFrame(events) if events else pd.DataFrame(columns=['time', 'spo2'])
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
        spo2 = self.df_spo2['spo2'].values
        times = self.df_spo2['time'].values
        n = len(spo2)
        assumed_duration = 60  # Assume events last ~60s

        events = []
        i = 0
        while i < n:
            # Rolling baseline: max SpO2 in preceding 120s
            base_start = max(0, i - 120)
            baseline = np.nanmax(spo2[base_start:i]) if i > 0 else spo2[0]

            if baseline - spo2[i] >= desat_threshold:
                start_time = times[i]
                end_time = min(start_time + assumed_duration, times[-1])
                events.append({"start": start_time, "end": end_time})
                # Skip past this event
                skip_to = i + assumed_duration
                i = min(int(skip_to), n)
            else:
                i += 1

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
        if use_mit_st and self.manual_stages is not None:
            logger.info("Using MIT gold standard annotations")
            self.stages = self.manual_stages
            return self.stages

        # Check if YASA is available and we have EEG
        if YASA_AVAILABLE:
            if not self.eeg_ch:
                logger.warning("YASA available but no EEG channel found - using rule-based staging")
            elif self.raw.info['sfreq'] < 100:
                logger.warning("YASA requires >=100 Hz, got %s Hz - using rule-based staging",
                               self.raw.info['sfreq'])
            else:
                logger.info("Attempting YASA deep learning staging (sfreq: %s Hz)",
                            self.raw.info['sfreq'])
                try:
                    stages = self._yasa_staging()
                    self.stages = stages
                    logger.info("YASA staging successful: %d epochs", len(stages))
                    return stages
                except Exception as e:
                    logger.error("YASA failed: %s: %s", type(e).__name__, e)
                    logger.warning("Falling back to rule-based staging...")
        else:
            logger.info("YASA not installed - using rule-based staging")

        # Fall back to rule-based staging
        logger.info("Using rule-based spectral staging")
        stages = self._rule_based_staging()
        self.stages = stages
        return stages
    
    def _yasa_staging(self):
        """
        Perform automatic sleep staging using YASA
        """
        # Check sampling rate
        if self.raw.info['sfreq'] < 100:
            raise ValueError(f"Sampling rate too low: {self.raw.info['sfreq']} Hz")

        logger.info("Attempting YASA staging with EEG: %s, EOG: %s, EMG: %s",
                     self.eeg_ch, self.eog_ch, self.emg_ch)

        sls = yasa.SleepStaging(
            self.raw,
            eeg_name=self.eeg_ch,
            eog_name=self.eog_ch,
            emg_name=self.emg_ch
        )

        logger.info("YASA SleepStaging object created successfully")

        hypno = sls.predict()
        logger.info("YASA prediction successful: %d epochs", len(hypno))

        # Convert YASA stages to standard format
        stage_map = {'W': 'W', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'R': 'REM'}
        stages = [stage_map.get(s, 'Unknown') for s in hypno]

        # Crop to recording duration
        max_epochs = int(self.raw.times[-1] / 30)
        stages = stages[:max_epochs]

        logger.info("YASA staging complete: %d epochs converted", len(stages))
        return stages
    
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
        except Exception as e:
            logger.warning("Could not create MNE Epochs for staging: %s", e)
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

            except Exception as e:
                logger.debug("Could not stage epoch %d: %s", i, e)
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
                    
                    # Calculate ODI for this stage using all epochs of this stage
                    stage_epoch_indices = [i for i, s in enumerate(self.stages) if s == stage]
                    odi_count = 0
                    for ei in stage_epoch_indices:
                        epoch_start = ei * 30
                        epoch_end = (ei + 1) * 30
                        odi_in_epoch = self.odi_events[
                            (self.odi_events['time'] >= epoch_start) &
                            (self.odi_events['time'] < epoch_end)
                        ]
                        odi_count += len(odi_in_epoch)
                    odi_stage = odi_count / hrs if hrs > 0 else 0

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
        
        # Assign each event a sleep stage based on its start time epoch
        event_stages = []
        for _, ev in self.events_df.iterrows():
            epoch_idx = int(ev['start'] // 30)
            if epoch_idx < len(self.stages):
                s = self.stages[epoch_idx]
                if s not in ('W', 'N1', 'N2', 'N3', 'REM'):
                    s = 'Unknown'
            else:
                s = 'Unknown'
            event_stages.append(s)
        self.events_df['stage'] = event_stages

        # Organize events by sleep stage
        stage_events = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': [], 'Unknown': [], 'Total': []}

        for stage in stage_events:
            if stage == 'Total':
                stage_events['Total'] = self.events_df.to_dict('records')
            else:
                evs = self.events_df[self.events_df['stage'] == stage]
                stage_events[stage] = evs.to_dict('records')
        
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
            
            # Calculate stage-specific ODI across all epochs of this stage
            stage_epoch_indices = [i for i, s in enumerate(self.stages) if s == stage]
            odi_count = 0
            for ei in stage_epoch_indices:
                epoch_start = ei * 30
                epoch_end = (ei + 1) * 30
                odi_in_epoch = self.odi_events[
                    (self.odi_events['time'] >= epoch_start) &
                    (self.odi_events['time'] < epoch_end)
                ]
                odi_count += len(odi_in_epoch)
            odi_stage = odi_count / hrs if hrs > 0 else 0
            
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
        
        Azarbarzin et al. (2019) methodology:
        - Baseline: 98th percentile of SpO2 in 100s BEFORE event START
        - Desaturation window: FROM event START to 90s AFTER event END
        
        Parameters:
        -----------
        events : list
            Events to analyze
        pre_event_sec : int
            Seconds before event START to measure baseline (default: 100)
        desat_start_sec : int
            Offset from event START (should be 0 for Azarbarzin method)
        desat_end_sec : int
            Seconds after event END to search for nadir (default: 90)
        artifact_filter : str
            Artifact filter setting
        """
        area_total = 0.0
        proof_events = []
        
        for ev in events:
            start_t = ev['start']  # Event START time (key change!)
            end_t = ev['end']      # Event END time
            
            # Get baseline from pre-event window (BEFORE event starts)
            # Azarbarzin: 100 seconds before event START
            base_df = self.df_spo2[
                (self.df_spo2['time'] >= start_t - pre_event_sec) &
                (self.df_spo2['time'] < start_t)
            ]
            
            if len(base_df) == 0:
                continue
            
            # Use MAXIMUM SpO2 for baseline (per Azarbarzin paper)
            # "the maximum SpO2 value during a 100-second search window 
            # immediately preceding the event"
            baseline = base_df['spo2'].max()
            
            # Get desaturation window
            # Azarbarzin: FROM event start TO 90s after event end
            # desat_start_sec allows offset from START (0 for exact Azarbarzin)
            win_df = self.df_spo2[
                (self.df_spo2['time'] >= start_t + desat_start_sec) &
                (self.df_spo2['time'] <= end_t + desat_end_sec)
            ].copy()
            
            if len(win_df) < 2:
                continue
            
            # Remove artifacts if artifact column exists
            if 'artifact' in self.df_spo2.columns:
                artifact_indices = self.df_spo2[self.df_spo2['artifact']].index
                win_df = win_df[~win_df.index.isin(artifact_indices)]
            
            # Calculate area below baseline (trapezoidal integration)
            depth = np.maximum(baseline - win_df['spo2'].values, 0)
            area = trapz(depth, win_df['time'].values)
            
            area_total += area
            
            # Store for proof plots
            proof_events.append({
                'start_t': start_t,  # Also store start time
                'end_t': end_t,
                'baseline': baseline,
                'win_df': win_df,
                'depth': depth,
                'area': area,
                'hb_contrib': area / 60
            })
        
        return area_total, proof_events
    
    def _get_event_free_epochs(self):
        """
        Identify 30s epochs with no overlapping respiratory events.

        An epoch is "event-free" if no detected apnea/hypopnea event's
        [start, end] interval overlaps with the epoch's [epoch_start, epoch_end].

        Returns:
        --------
        np.ndarray (bool)
            Array of length n_epochs, True for event-free epochs
        """
        n_epochs = len(self.stages) if self.stages else int(self.df_spo2['time'].max() // 30)
        event_free = np.ones(n_epochs, dtype=bool)

        if self.events_df is not None and len(self.events_df) > 0:
            for _, ev in self.events_df.iterrows():
                # Mark all epochs overlapping this event as NOT event-free
                ev_start_epoch = int(ev['start'] // 30)
                ev_end_epoch = int(ev['end'] // 30)
                for ei in range(max(0, ev_start_epoch), min(n_epochs, ev_end_epoch + 1)):
                    event_free[ei] = False

        return event_free

    def _calculate_floating_baseline(self, window_sec=600, percentile=95,
                                       lower_quartile_cutoff=25):
        """
        Calculate a floating (trailing) baseline for each SpO₂ sample.

        For each timepoint, the baseline is computed from the trailing
        window of SpO₂ values by:
        1. Removing the lowest quartile (likely desaturation values)
        2. Taking the Nth percentile of the remaining upper values

        This provides a locally-adaptive baseline that tracks physiological
        changes across sleep stages without requiring staging data.

        Parameters:
        -----------
        window_sec : int
            Trailing window duration in seconds (default 600 = 10 min)
        percentile : float
            Percentile to compute from the filtered window (default 95)
        lower_quartile_cutoff : float
            Bottom N% of values to exclude from each window (default 25)

        Returns:
        --------
        np.ndarray
            Floating baseline value for each SpO₂ sample
        """
        spo2_vals = self.df_spo2['spo2'].values
        n = len(spo2_vals)
        floating_bl = np.full(n, np.nan)

        for i in range(n):
            # Trailing window: [i - window_sec, i] at 1Hz
            start_idx = max(0, i - window_sec)
            window = spo2_vals[start_idx:i + 1]

            if len(window) < 10:
                # Too few samples — use what we have
                floating_bl[i] = np.nanpercentile(window, percentile)
                continue

            # Remove bottom quartile (desaturation contamination)
            cutoff_val = np.nanpercentile(window, lower_quartile_cutoff)
            upper_vals = window[window >= cutoff_val]

            if len(upper_vals) > 0:
                floating_bl[i] = np.nanpercentile(upper_vals, percentile)
            else:
                floating_bl[i] = np.nanpercentile(window, percentile)

        return floating_bl

    def calculate_global_hypoxic_burden(self, preset_baseline=0.0,
                                        baseline_method='stage_specific',
                                        floating_window_sec=600):
        """
        Calculate global hypoxic burden.

        Supports three baseline methods:
        - 'stage_specific' (default): 95th percentile from event-free epochs
          per sleep stage. Falls back to whole-night if no staging available.
        - 'floating': Trailing 10-min window with bottom quartile removed,
          then 95th percentile. Adapts locally without requiring staging.
        - 'whole_night': Single 95th percentile across the entire study.

        Parameters:
        -----------
        preset_baseline : float
            Manual baseline SpO₂. If > 0, overrides all methods.
        baseline_method : str
            'stage_specific', 'floating', or 'whole_night'
        floating_window_sec : int
            Window size for floating baseline (default 600 = 10 min)

        Returns:
        --------
        dict
            global_hb, baseline, area, stage_baselines, fallback_baseline,
            baseline_method, floating_baseline (array, if method='floating')
        """
        if self.df_spo2 is None:
            raise ValueError("Must run preprocess_spo2() first")

        spo2_vals = self.df_spo2['spo2'].values
        spo2_times = self.df_spo2['time'].values

        # Whole-night 95th percentile (always computed as reference/fallback)
        fallback_baseline = calculate_robust_baseline(
            spo2_vals, method='percentile', percentile=95
        )

        # If manual baseline is set, override everything
        if preset_baseline > 0:
            depth_global = np.maximum(preset_baseline - spo2_vals, 0)
            global_desat_area = trapz(depth_global, spo2_times)
            total_sleep_sec = spo2_times[-1] - spo2_times[0] if len(spo2_times) > 1 else spo2_times[-1]
            total_hours = total_sleep_sec / 3600
            global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0
            return {
                'global_hb': global_hb,
                'baseline': preset_baseline,
                'area': global_desat_area,
                'stage_baselines': {},
                'fallback_baseline': preset_baseline,
                'baseline_method': 'manual'
            }

        # ---- FLOATING BASELINE METHOD ----
        if baseline_method == 'floating':
            logger.info("Computing floating baseline (window=%ds)", floating_window_sec)
            floating_bl = self._calculate_floating_baseline(
                window_sec=floating_window_sec,
                percentile=95,
                lower_quartile_cutoff=25
            )

            # Integrate: max(floating_baseline[i] - spo2[i], 0) per sample
            depth_global = np.maximum(floating_bl - spo2_vals, 0)
            global_desat_area = trapz(depth_global, spo2_times)
            total_sleep_sec = spo2_times[-1] - spo2_times[0] if len(spo2_times) > 1 else spo2_times[-1]
            total_hours = total_sleep_sec / 3600
            global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0

            # Summary baseline = mean of the floating baseline array
            avg_bl = float(np.nanmean(floating_bl))

            return {
                'global_hb': global_hb,
                'baseline': avg_bl,
                'area': global_desat_area,
                'stage_baselines': {},
                'fallback_baseline': fallback_baseline,
                'baseline_method': 'floating',
                'floating_baseline': floating_bl,
                'floating_window_sec': floating_window_sec
            }

        # ---- WHOLE-NIGHT BASELINE METHOD ----
        if baseline_method == 'whole_night' or (
            self.stages is None or len(self.stages) == 0
        ):
            depth_global = np.maximum(fallback_baseline - spo2_vals, 0)
            global_desat_area = trapz(depth_global, spo2_times)
            total_sleep_sec = spo2_times[-1] - spo2_times[0] if len(spo2_times) > 1 else spo2_times[-1]
            total_hours = total_sleep_sec / 3600
            global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0
            return {
                'global_hb': global_hb,
                'baseline': fallback_baseline,
                'area': global_desat_area,
                'stage_baselines': {},
                'fallback_baseline': fallback_baseline,
                'baseline_method': 'whole_night'
            }

        # ---- STAGE-SPECIFIC BASELINE METHOD (default) ----
        event_free = self._get_event_free_epochs()
        stage_baselines = {}

        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            stage_ef_indices = [
                i for i, s in enumerate(self.stages)
                if s == stage and i < len(event_free) and event_free[i]
            ]

            if stage_ef_indices:
                ef_spo2 = []
                for ei in stage_ef_indices:
                    epoch_start = ei * 30
                    epoch_end = (ei + 1) * 30
                    mask = (spo2_times >= epoch_start) & (spo2_times < epoch_end)
                    ef_spo2.extend(spo2_vals[mask])

                if len(ef_spo2) > 0:
                    stage_baselines[stage] = calculate_robust_baseline(
                        np.array(ef_spo2), method='percentile', percentile=95
                    )
                else:
                    stage_baselines[stage] = fallback_baseline
            else:
                stage_baselines[stage] = fallback_baseline

        # Integrate per-sample using stage-appropriate baseline
        depth_global = np.zeros_like(spo2_vals, dtype=float)
        n_stages = len(self.stages)

        for i, t in enumerate(spo2_times):
            epoch_idx = min(int(t // 30), n_stages - 1)
            if epoch_idx >= 0 and epoch_idx < n_stages:
                stage = self.stages[epoch_idx]
            else:
                stage = 'Unknown'
            bl = stage_baselines.get(stage, fallback_baseline)
            depth_global[i] = max(bl - spo2_vals[i], 0)

        global_desat_area = trapz(depth_global, spo2_times)
        total_sleep_sec = spo2_times[-1] - spo2_times[0] if len(spo2_times) > 1 else spo2_times[-1]

        total_hours = total_sleep_sec / 3600
        global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0

        # Weighted average baseline for summary display
        stage_counts = pd.Series(self.stages).value_counts()
        total_staged_epochs = sum(stage_counts.get(s, 0) for s in stage_baselines)
        if total_staged_epochs > 0:
            weighted_bl = sum(
                stage_baselines.get(s, fallback_baseline) * stage_counts.get(s, 0)
                for s in stage_baselines
            ) / total_staged_epochs
        else:
            weighted_bl = fallback_baseline

        return {
            'global_hb': global_hb,
            'baseline': weighted_bl,
            'area': global_desat_area,
            'stage_baselines': stage_baselines,
            'fallback_baseline': fallback_baseline,
            'baseline_method': 'stage_specific'
        }
    
    def calculate_bootstrap_ci(self, n_boot=1000, pre_event_sec=100,
                               desat_start_sec=0, desat_end_sec=90,
                               artifact_filter='Off', stratify_by_stage=True):
        """
        Calculate 95% confidence interval using bootstrap
        
        Uses Azarbarzin methodology:
        - Baseline from 100s before event START
        - MAXIMUM baseline value
        - Desaturation window from event START to 90s after END
        
        Parameters:
        -----------
        n_boot : int
            Number of bootstrap iterations (default: 1000)
        pre_event_sec : int
            Seconds before event START for baseline (default: 100)
        desat_start_sec : int
            Offset from event START (default: 0 for Azarbarzin)
        desat_end_sec : int
            Seconds after event END (default: 90)
        artifact_filter : str
            Artifact filter setting
        stratify_by_stage : bool
            If True, maintain stage distribution in bootstrap samples
            (more accurate CI, especially for small samples)
        
        Returns:
        --------
        tuple
            (ci_low, ci_high)
        
        Notes:
        ------
        Assumptions:
        - Events within stages are independent
        - TST is fixed (appropriate for single-recording analysis)
        - Minimum 20 events recommended for reliable CI
        
        Stratification:
        - Preserves sleep stage distribution across bootstrap samples
        - Reduces variance from stage composition changes
        - Recommended for most analyses
        """
        if self.events_df is None or len(self.events_df) == 0:
            return (0.0, 0.0)
        
        n_events = len(self.events_df)
        
        # Warn for small samples
        if n_events < 20:
            import warnings
            warnings.warn(
                f"Only {n_events} events detected. Bootstrap 95% CI may be "
                f"unreliable with fewer than 20 events. Interpret with caution. "
                f"Consider wider confidence intervals in practice.",
                UserWarning
            )
        
        hb_values = []
        total_hours = self.raw.times[-1] / 3600
        
        for _ in range(n_boot):
            # Bootstrap sample events
            if stratify_by_stage and self.stages is not None:
                # Stage-stratified bootstrap
                boot_events_list = []
                
                # Sample within each stage to preserve distribution
                for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                    stage_events = self.events_df[self.events_df['stage'] == stage]
                    
                    if len(stage_events) > 0:
                        # Sample with replacement within this stage
                        boot_stage = stage_events.sample(
                            n=len(stage_events), 
                            replace=True
                        )
                        boot_events_list.append(boot_stage)
                
                # Combine all stages
                if boot_events_list:
                    boot_events = pd.concat(boot_events_list, ignore_index=True)
                else:
                    # Fallback to unstratified if no stage info
                    boot_events = self.events_df.sample(n=n_events, replace=True)
            else:
                # Simple (unstratified) bootstrap
                boot_events = self.events_df.sample(n=n_events, replace=True)
            
            area_total = 0.0
            
            for _, ev in boot_events.iterrows():
                start_t = ev['start']
                end_t = ev['end']
                
                # Get baseline (100s before event START, MAXIMUM value)
                base_df = self.df_spo2[
                    (self.df_spo2['time'] >= start_t - pre_event_sec) &
                    (self.df_spo2['time'] < start_t)
                ]
                
                if len(base_df) == 0:
                    continue
                
                baseline = base_df['spo2'].max()
                
                # Get desaturation window (from START to END+90s)
                win_df = self.df_spo2[
                    (self.df_spo2['time'] >= start_t + desat_start_sec) &
                    (self.df_spo2['time'] <= end_t + desat_end_sec)
                ]
                
                if len(win_df) < 2:
                    continue
                
                # Remove artifacts if artifact column exists
                if 'artifact' in self.df_spo2.columns:
                    artifact_indices = self.df_spo2[self.df_spo2['artifact']].index
                    win_df = win_df[~win_df.index.isin(artifact_indices)]
                
                # Calculate area
                depth = np.maximum(baseline - win_df['spo2'].values, 0)
                area = trapz(depth, win_df['time'].values)
                area_total += area
            
            # HB for this bootstrap sample
            hb_boot = (area_total / 60) / total_hours
            hb_values.append(hb_boot)
        
        # Calculate 95% CI using percentile method
        ci_low, ci_high = np.percentile(hb_values, [2.5, 97.5])
        
        return (ci_low, ci_high)
    
    def run_full_analysis(self, pre_event_sec=100, desat_start_sec=0,
                         desat_end_sec=90, artifact_filter='Off',
                         desat_threshold=3, use_global_hb=True,
                         preset_baseline=0.0, use_mit_st=False,
                         baseline_method='stage_specific',
                         floating_window_sec=600):
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
                n_boot=1000,
                pre_event_sec=pre_event_sec,
                desat_start_sec=desat_start_sec,
                desat_end_sec=desat_end_sec,
                artifact_filter=artifact_filter,
                stratify_by_stage=True  # Use stage-stratified bootstrap for more accurate CI
            )
        else:
            ci_low, ci_high = 0.0, 0.0
        
        # Step 7: Calculate global HB (if requested)
        global_hb_results = None
        if use_global_hb:
            global_hb_results = self.calculate_global_hypoxic_burden(
                preset_baseline=preset_baseline,
                baseline_method=baseline_method,
                floating_window_sec=floating_window_sec
            )
        
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
            results['stage_baselines'] = global_hb_results.get('stage_baselines', {})
            results['fallback_baseline'] = global_hb_results.get('fallback_baseline', 0)
            results['baseline_method'] = global_hb_results.get('baseline_method', 'stage_specific')
            if 'floating_baseline' in global_hb_results:
                results['floating_baseline'] = global_hb_results['floating_baseline']
                results['floating_window_sec'] = global_hb_results.get('floating_window_sec', 600)

        return results

    # ==================================================================
    # PAREKH 2023 METHOD — Automated SpO₂ nadir detection
    # Parekh A et al. AJRCCM 2023;208(11):1216-1226
    # ==================================================================

    def _parekh_detect_desaturations(self, min_prominence=3.0, min_duration=4):
        """
        Detect desaturation events using Parekh peak-prominence method.

        Identifies SpO₂ nadirs with ≥3% prominence and their flanking
        left/right peaks. Does NOT require manually-scored respiratory
        events — works directly on the SpO₂ signal.

        Parameters:
        -----------
        min_prominence : float
            Minimum SpO₂ drop (%) for a nadir to qualify (default 3.0)
        min_duration : float
            Minimum event duration in seconds (default 4)

        Returns:
        --------
        list of dict
            Each dict contains: nadir_idx, nadir_time, nadir_spo2,
            left_peak_idx, left_peak_time, left_peak_spo2,
            right_peak_idx, right_peak_time, right_peak_spo2
        """
        from scipy.signal import find_peaks, savgol_filter

        spo2 = self.df_spo2['spo2'].values.copy()
        times = self.df_spo2['time'].values

        # Savitzky-Golay smoothing (window=11s at 1Hz, polyorder=3)
        # Creates an interpolated continuous SpO₂ signal per Parekh method
        win_len = min(11, len(spo2))
        if win_len % 2 == 0:
            win_len -= 1  # Must be odd
        if win_len >= 5:
            spo2_smooth = savgol_filter(spo2, window_length=win_len, polyorder=3)
        else:
            spo2_smooth = spo2.copy()

        # Find nadirs: invert SpO₂ and find peaks with ≥ min_prominence
        inverted = -spo2_smooth
        peaks, properties = find_peaks(
            inverted,
            prominence=min_prominence,
            distance=10  # At least 10s between nadirs
        )

        if len(peaks) == 0:
            logger.info("Parekh: no desaturation nadirs found with ≥%.1f%% prominence",
                        min_prominence)
            return []

        events = []
        for i, nadir_idx in enumerate(peaks):
            # Left peak: max SpO₂ between previous nadir (or signal start) and this nadir
            left_bound = peaks[i - 1] if i > 0 else 0
            left_region = spo2_smooth[left_bound:nadir_idx]
            if len(left_region) == 0:
                continue
            left_peak_local = np.argmax(left_region)
            left_peak_idx = left_bound + left_peak_local

            # Right peak: max SpO₂ between this nadir and next nadir (or signal end)
            right_bound = peaks[i + 1] if i < len(peaks) - 1 else len(spo2_smooth)
            right_region = spo2_smooth[nadir_idx:right_bound]
            if len(right_region) == 0:
                continue
            right_peak_local = np.argmax(right_region)
            right_peak_idx = nadir_idx + right_peak_local

            # Duration check (left peak → right peak)
            duration = times[right_peak_idx] - times[left_peak_idx]
            if duration < min_duration:
                continue

            events.append({
                'nadir_idx': nadir_idx,
                'nadir_time': times[nadir_idx],
                'nadir_spo2': spo2_smooth[nadir_idx],
                'left_peak_idx': left_peak_idx,
                'left_peak_time': times[left_peak_idx],
                'left_peak_spo2': spo2_smooth[left_peak_idx],
                'right_peak_idx': right_peak_idx,
                'right_peak_time': times[right_peak_idx],
                'right_peak_spo2': spo2_smooth[right_peak_idx],
            })

        logger.info("Parekh: detected %d desaturation events (≥%.1f%% prominence)",
                     len(events), min_prominence)
        return events

    def _parekh_calculate_hb(self, parekh_events):
        """
        Calculate HB using Parekh method: area between the line
        connecting left/right peaks and the SpO₂ trace below it.

        Parameters:
        -----------
        parekh_events : list of dict
            Events from _parekh_detect_desaturations()

        Returns:
        --------
        tuple (float, list)
            total_area : total desaturation area (%-seconds)
            proof_events : list of event dicts for visualization
        """
        spo2 = self.df_spo2['spo2'].values
        times = self.df_spo2['time'].values
        total_area = 0.0
        proof_events = []

        for ev in parekh_events:
            li = ev['left_peak_idx']
            ri = ev['right_peak_idx']

            if ri <= li or ri >= len(spo2):
                continue

            segment_times = times[li:ri + 1]
            segment_spo2 = spo2[li:ri + 1]

            if len(segment_spo2) < 2:
                continue

            # Baseline: linear interpolation between left and right peaks
            left_spo2 = ev['left_peak_spo2']
            right_spo2 = ev['right_peak_spo2']
            baseline_line = np.linspace(left_spo2, right_spo2, len(segment_spo2))

            # Area = integral of max(baseline_line - SpO₂, 0)
            depth = np.maximum(baseline_line - segment_spo2, 0)
            area = trapz(depth, segment_times)
            total_area += area

            # Average baseline for this event (midpoint of peak line)
            avg_baseline = (left_spo2 + right_spo2) / 2.0

            # Store for proof plots (compatible with existing proof event format)
            win_df = self.df_spo2[
                (self.df_spo2['time'] >= times[li]) &
                (self.df_spo2['time'] <= times[ri])
            ].copy()

            proof_events.append({
                'start_t': ev['left_peak_time'],
                'end_t': ev['right_peak_time'],
                'baseline': avg_baseline,
                'win_df': win_df,
                'depth': depth,
                'area': area,
                'hb_contrib': area / 60
            })

        return total_area, proof_events

    def _parekh_bootstrap_ci(self, parekh_events, n_boot=1000):
        """
        Calculate 95% confidence interval for Parekh HB via bootstrap.

        Resamples the Parekh desaturation events and recomputes HB.

        Parameters:
        -----------
        parekh_events : list of dict
            Events from _parekh_detect_desaturations()
        n_boot : int
            Number of bootstrap iterations

        Returns:
        --------
        tuple (float, float)
            (ci_low, ci_high) at 2.5th and 97.5th percentiles
        """
        if len(parekh_events) == 0:
            return (0.0, 0.0)

        spo2 = self.df_spo2['spo2'].values
        times = self.df_spo2['time'].values
        total_hours = (times[-1] - times[0]) / 3600 if len(times) > 1 else times[-1] / 3600

        if total_hours <= 0:
            return (0.0, 0.0)

        hb_values = []
        n_events = len(parekh_events)

        for _ in range(n_boot):
            # Resample events with replacement
            boot_indices = np.random.choice(n_events, size=n_events, replace=True)
            boot_area = 0.0

            for idx in boot_indices:
                ev = parekh_events[idx]
                li = ev['left_peak_idx']
                ri = ev['right_peak_idx']

                if ri <= li or ri >= len(spo2):
                    continue

                segment_spo2 = spo2[li:ri + 1]
                segment_times = times[li:ri + 1]

                if len(segment_spo2) < 2:
                    continue

                baseline_line = np.linspace(
                    ev['left_peak_spo2'], ev['right_peak_spo2'],
                    len(segment_spo2)
                )
                depth = np.maximum(baseline_line - segment_spo2, 0)
                boot_area += trapz(depth, segment_times)

            hb_boot = boot_area / 60 / total_hours
            hb_values.append(hb_boot)

        ci_low, ci_high = np.percentile(hb_values, [2.5, 97.5])
        return (ci_low, ci_high)

    def run_parekh_analysis(self, artifact_filter='Off', desat_threshold=3,
                            use_global_hb=True, use_mit_st=False,
                            baseline_method='stage_specific',
                            floating_window_sec=600):
        """
        Run complete Parekh 2023 analysis pipeline.

        Uses automated SpO₂ nadir detection (≥3% peak prominence) with
        left/right peak baselines instead of manually-scored respiratory
        events with pre-event time windows.

        Parameters:
        -----------
        artifact_filter : str
            SpO₂ artifact filter setting
        desat_threshold : int
            Desaturation threshold for ODI detection (3 or 4%)
        use_global_hb : bool
            Also calculate global HB
        use_mit_st : bool
            Use MIT annotations for staging if available

        Returns:
        --------
        dict
            Results in the same format as run_full_analysis()
        """
        # Step 1: Preprocess SpO₂ (reuse existing)
        self.preprocess_spo2(artifact_filter)

        # Step 2: Detect ODI events (for ODI metric display)
        self.detect_odi_events(desat_threshold)

        # Step 3: Sleep staging (for stage-specific metrics)
        self.perform_sleep_staging(use_mit_st)

        # Step 4: Parekh desaturation detection
        parekh_events = self._parekh_detect_desaturations(
            min_prominence=float(desat_threshold),
            min_duration=4
        )

        # Step 5: Calculate HB from Parekh events
        total_area, proof_events = self._parekh_calculate_hb(parekh_events)

        # Step 6: Calculate total hours and HB
        total_hours = self.raw.times[-1] / 3600
        total_hb = (total_area / 60) / total_hours if total_hours > 0 else 0

        # Step 7: Assign Parekh events to sleep stages for stage-specific metrics
        # Convert Parekh events to events_df format for compatibility
        event_records = []
        for ev in parekh_events:
            event_records.append({
                'start': ev['left_peak_time'],
                'end': ev['right_peak_time']
            })
        self.events_df = pd.DataFrame(event_records) if event_records else pd.DataFrame(columns=['start', 'end'])

        # Assign stages to events
        stage_events = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': [], 'Unknown': [], 'Total': []}

        if len(self.events_df) > 0 and self.stages is not None:
            event_stages = []
            for _, ev in self.events_df.iterrows():
                epoch_idx = int(ev['start'] // 30)
                if epoch_idx < len(self.stages):
                    s = self.stages[epoch_idx]
                    if s not in ('W', 'N1', 'N2', 'N3', 'REM'):
                        s = 'Unknown'
                else:
                    s = 'Unknown'
                event_stages.append(s)
            self.events_df['stage'] = event_stages

            for stage in stage_events:
                if stage == 'Total':
                    stage_events['Total'] = self.events_df.to_dict('records')
                else:
                    evs = self.events_df[self.events_df['stage'] == stage]
                    stage_events[stage] = evs.to_dict('records')

        # Calculate stage-specific metrics
        stage_results = {}
        if self.stages is not None:
            stage_counts = pd.Series(self.stages).value_counts()
            stage_time = (stage_counts * 30 / 3600).to_dict()

            # Group Parekh events by stage for HB calculation
            parekh_by_stage = {}
            for pev in parekh_events:
                epoch_idx = int(pev['nadir_time'] // 30)
                if self.stages is not None and epoch_idx < len(self.stages):
                    s = self.stages[epoch_idx]
                    if s not in ('W', 'N1', 'N2', 'N3', 'REM'):
                        s = 'Unknown'
                else:
                    s = 'Unknown'
                parekh_by_stage.setdefault(s, []).append(pev)

            total_hb_weighted = 0.0
            total_time = 0.0

            for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                hrs = stage_time.get(stage, 0)
                if hrs == 0:
                    continue

                # Parekh HB for this stage
                stage_pevents = parekh_by_stage.get(stage, [])
                stage_area, _ = self._parekh_calculate_hb(stage_pevents)
                hb_stage = (stage_area / 60) / hrs if hrs > 0 else 0

                # ODI for this stage
                stage_epoch_indices = [i for i, s in enumerate(self.stages) if s == stage]
                odi_count = 0
                for ei in stage_epoch_indices:
                    epoch_start = ei * 30
                    epoch_end = (ei + 1) * 30
                    odi_in_epoch = self.odi_events[
                        (self.odi_events['time'] >= epoch_start) &
                        (self.odi_events['time'] < epoch_end)
                    ]
                    odi_count += len(odi_in_epoch)
                odi_stage = odi_count / hrs if hrs > 0 else 0

                n_stage_events = len(stage_pevents)
                ahi_stage = n_stage_events / hrs if hrs > 0 else 0

                stage_results[stage] = {
                    'hrs': hrs,
                    'AHI': ahi_stage,
                    'ODI': odi_stage,
                    'HB': hb_stage
                }

                total_hb_weighted += hb_stage * hrs
                total_time += hrs

            # Recalculate total_hb as weighted average if stages available
            if total_time > 0:
                total_hb = total_hb_weighted / total_time

        # Step 8: Bootstrap CI
        if len(parekh_events) > 0:
            ci_low, ci_high = self._parekh_bootstrap_ci(parekh_events, n_boot=1000)
        else:
            ci_low, ci_high = 0.0, 0.0

        # Step 9: Calculate global HB
        global_hb_results = None
        if use_global_hb:
            global_hb_results = self.calculate_global_hypoxic_burden(
                baseline_method=baseline_method,
                floating_window_sec=floating_window_sec
            )

        # Step 10: Compile metrics
        ahi = len(parekh_events) / total_hours if total_hours > 0 else 0
        odi = len(self.odi_events) / total_hours if total_hours > 0 else 0

        # Compile results (same format as run_full_analysis)
        results = {
            'duration': total_hours,
            'ahi': ahi,
            'odi': odi,
            'total_hb': total_hb,
            'ci': (ci_low, ci_high),
            'events': proof_events,
            'stage_hb': stage_results,
            'manual_ahi': self.manual_ahi,
            'use_mit_st': use_mit_st and self.manual_stages is not None,
            'method': 'parekh',
            'parekh_events_count': len(parekh_events)
        }

        # Add global HB if calculated
        if global_hb_results:
            results['global_hb'] = global_hb_results['global_hb']
            results['baseline_used'] = global_hb_results['baseline']
            results['stage_baselines'] = global_hb_results.get('stage_baselines', {})
            results['fallback_baseline'] = global_hb_results.get('fallback_baseline', 0)
            results['baseline_method'] = global_hb_results.get('baseline_method', 'stage_specific')
            if 'floating_baseline' in global_hb_results:
                results['floating_baseline'] = global_hb_results['floating_baseline']
                results['floating_window_sec'] = global_hb_results.get('floating_window_sec', 600)

        return results
