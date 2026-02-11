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
        
        # Calculate area below baseline (integral of max(baseline - SpO₂, 0))
        depth_global = np.maximum(baseline - self.df_spo2['spo2'].values, 0)
        global_desat_area = trapz(depth_global, self.df_spo2['time'].values)

        total_sleep_sec = self.df_spo2['time'].max()
        
        # Convert to (%min)/h
        total_hours = total_sleep_sec / 3600
        global_hb = global_desat_area / 60 / total_hours if total_hours > 0 else 0
        
        return {
            'global_hb': global_hb,
            'baseline': baseline,
            'area': global_desat_area
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
