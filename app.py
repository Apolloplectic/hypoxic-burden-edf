"""
Hypoxic Burden Calculator - Polysomnography Analysis Tool
Based on: Azarbarzin A, et al. European Heart Journal (2019)
DOI: 10.1093/eurheartj/ehy624

Author: Sam Johnson
Email: sam.johnson9797@gmail.com
GitHub: https://github.com/Apolloplectic/hypoxic-burden-edf
"""

import streamlit as st
import os
from datetime import datetime
import io
import zipfile

# Import custom modules
from analysis_engine import PSGAnalyzer
from pdf_generator import PDFReportGenerator
from utils import initialize_session_state, load_edf_file
from config import YASA_AVAILABLE

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(
    page_title="Hypoxic Burden Calculator",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.title("🫁 Hypoxic Burden Calculator")
st.markdown("""
**Upload PSG EDF file** → get comprehensive sleep apnea metrics with **95% CI**.

Based on:
> Azarbarzin A, et al. *European Heart Journal* (2019) – DOI: [10.1093/eurheartj/ehy624](https://doi.org/10.1093/eurheartj/ehy624)
""")

# --------------------------------------------------------------
# FILE UPLOAD SECTION
# --------------------------------------------------------------
st.markdown("### 📁 Single File Analysis")
edf_file = st.file_uploader(
    "Upload PSG EDF file",
    type=["edf"],
    help="⚠️ Online version limited to 200 MB. For larger files (up to 2 GB), run locally (see instructions below).",
    key="single_file_upload"
)

# --------------------------------------------------------------
# LOCAL RUN INSTRUCTIONS
# --------------------------------------------------------------
with st.expander("📥 File too large? Run locally (2 GB+ support) — no coding needed!", expanded=False):
    st.markdown("""
    ### **How to Run This App on Your Computer (2 GB+ Files)**
    **No programming experience required. Takes 5 minutes.**
    
    ---
    
    #### **Step 1: Install Python (if not already installed)**
    1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
    2. During installation, **check "Add Python to PATH"**
    
    ---
    
    #### **Step 2: Download & Setup**
    1. Download the app from [GitHub Releases](https://github.com/Apolloplectic/hypoxic-burden-edf/releases)
    2. Unzip the folder
    3. Open terminal/command prompt in that folder
    4. Run: `pip install -r requirements.txt`
    
    ---
    
    #### **Step 3: Run the App**
    In terminal, run: `streamlit run app.py --server.maxUploadSize=4096`
    
    The `--server.maxUploadSize=4096` flag allows uploads up to **4 GB**.
    
    Your browser will open automatically with the app running locally.
    
    ---
    
    **Need help?** 
    - 📧 Email: `sam.johnson9797@gmail.com`
    - 🐙 GitHub Issues: [Report a problem](https://github.com/Apolloplectic/hypoxic-burden-edf/issues)
    """)

# --------------------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------------------
initialize_session_state()

# --------------------------------------------------------------
# SETTING PRESETS
# --------------------------------------------------------------
PRESETS = {
    "Azarbarzin 2019 (Default)": {
        'pre_event_sec': 100,
        'desat_start_sec': 60,
        'desat_end_sec': 120,
        'desat_threshold': 3,
        'artifact_filter': 'Off',
        'use_global_hb': True,
        'preset_baseline': 0.0,
        'description': "Validated parameters from the original Azarbarzin et al. study (EHJ 2019)"
    },
    "AASM 2023 Standard": {
        'pre_event_sec': 120,
        'desat_start_sec': 60,
        'desat_end_sec': 90,
        'desat_threshold': 3,
        'artifact_filter': 'Mild (10%/s)',
        'use_global_hb': True,
        'preset_baseline': 0.0,
        'description': "Current clinical practice guidelines with conservative parameters"
    },
    "Conservative (High Specificity)": {
        'pre_event_sec': 100,
        'desat_start_sec': 60,
        'desat_end_sec': 120,
        'desat_threshold': 4,
        'artifact_filter': 'Strict (5%/s)',
        'use_global_hb': False,
        'preset_baseline': 0.0,
        'description': "Minimizes false positives - only counts definite events (4% threshold)"
    },
    "Aggressive (High Sensitivity)": {
        'pre_event_sec': 80,
        'desat_start_sec': 45,
        'desat_end_sec': 150,
        'desat_threshold': 3,
        'artifact_filter': 'Off',
        'use_global_hb': True,
        'preset_baseline': 0.0,
        'description': "Maximizes event detection - catches all possible desaturations"
    },
    "Custom": {
        'pre_event_sec': 100,
        'desat_start_sec': 60,
        'desat_end_sec': 120,
        'desat_threshold': 3,
        'artifact_filter': 'Off',
        'use_global_hb': True,
        'preset_baseline': 0.0,
        'description': "Manually configure all parameters using sliders"
    }
}

# =============================================
# SINGLE FILE ANALYSIS MODE
# =============================================
if edf_file is not None:
    # Load EDF file
    with st.spinner(f"📂 Loading {edf_file.name} ({edf_file.size / 1e6:.1f} MB)..."):
        raw, temp_path = load_edf_file(edf_file)
    
    if raw is None:
        st.error("❌ Failed to load EDF file. Please check the file format.")
        st.stop()
    
    st.success(f"✅ EDF loaded successfully! Duration: {raw.times[-1]/3600:.2f} hours")
    
    # Initialize analyzer
    analyzer = PSGAnalyzer(raw, temp_path)
    
    # Display detected channels
    st.write(f"**SpO₂:** `{analyzer.spo2_ch or 'Not found ❌'}`")
    st.write(f"**Airflow:** `{analyzer.flow_ch or 'Not found (will use SpO₂-based detection)'}`")
    st.write(f"**EEG:** `{analyzer.eeg_ch or 'Not found (staging limited)'}`")
    
    if not analyzer.spo2_ch:
        st.error("❌ SpO₂ channel is required for analysis.")
        st.stop()
    
    # Check for MIT annotations
    if analyzer.check_mit_annotations():
        use_mit = st.checkbox(
            "✨ Use MIT Gold Standard Annotations",
            value=True,
            help="MIT-annotated sleep stages and events from SHHS/slpdb database"
        )
        st.session_state.use_mit_st = use_mit
        if use_mit:
            st.success(f"🎯 MIT annotations loaded: {len(analyzer.manual_events)} events, AHI = {analyzer.manual_ahi:.1f}")
    else:
        st.info("ℹ️ No MIT annotations found — using automated detection")
        st.session_state.use_mit_st = False
    
    # --------------------------------------------------------------
    # ADVANCED SETTINGS WITH PRESETS
    # --------------------------------------------------------------
    with st.expander("⚙️ Advanced Settings", expanded=False):
        # Preset selector
        preset_choice = st.selectbox(
            "📋 Analysis Preset",
            list(PRESETS.keys()),
            index=0,
            help="Choose a validated preset or customize all parameters"
        )
        
        # Show preset description
        st.info(f"ℹ️ **{preset_choice}:** {PRESETS[preset_choice]['description']}")
        
        # Load preset values
        params = PRESETS[preset_choice].copy()
        
        st.markdown("---")
        
        # If Custom, show sliders; otherwise show read-only values
        if preset_choice == "Custom":
            st.markdown("#### 🎚️ Event Detection Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                params['pre_event_sec'] = st.slider(
                    "Pre-event baseline window (s)",
                    min_value=30,
                    max_value=180,
                    value=params['pre_event_sec'],
                    step=1,
                    help="Time before event to calculate baseline SpO₂"
                )
                params['desat_start_sec'] = st.slider(
                    "Desaturation start (s before event end)",
                    min_value=15,
                    max_value=120,
                    value=params['desat_start_sec'],
                    step=1,
                    help="Start of desaturation window"
                )
            
            with col2:
                params['desat_end_sec'] = st.slider(
                    "Desaturation end (s after event end)",
                    min_value=60,
                    max_value=240,
                    value=params['desat_end_sec'],
                    step=1,
                    help="End of desaturation window (recovery time)"
                )
                params['artifact_filter'] = st.selectbox(
                    "SpO₂ artifact filter",
                    ["Off", "Mild (10%/s)", "Strict (5%/s)"],
                    index=["Off", "Mild (10%/s)", "Strict (5%/s)"].index(params['artifact_filter']),
                    help="Remove physiologically impossible SpO₂ changes"
                )
            
            st.markdown("#### 📊 Scoring Parameters")
            col3, col4 = st.columns(2)
            
            with col3:
                scoring_rule = st.selectbox(
                    "Desaturation threshold",
                    ["3% (AASM)", "4% (Legacy)"],
                    index=0 if params['desat_threshold'] == 3 else 1,
                    help="AASM recommends 3%"
                )
                params['desat_threshold'] = 3 if "3%" in scoring_rule else 4
            
            with col4:
                params['use_global_hb'] = st.checkbox(
                    "Calculate Global HB",
                    value=params['use_global_hb'],
                    help="Total oxygen debt over entire study"
                )
            
            if params['use_global_hb']:
                baseline_method = st.radio(
                    "Baseline SpO₂ method",
                    ["Automatic (95th percentile)", "Manual entry"],
                    help="Auto removes outliers/desaturations"
                )
                
                if baseline_method == "Manual entry":
                    params['preset_baseline'] = st.slider(
                        "Baseline SpO₂ (%)",
                        min_value=80.0,
                        max_value=100.0,
                        value=95.0,
                        step=0.1,
                        format="%.1f"
                    )
                else:
                    params['preset_baseline'] = 0.0
            else:
                params['preset_baseline'] = 0.0
        
        else:
            # Show preset values (read-only)
            st.markdown("#### 📋 Preset Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Pre-event baseline:** {params['pre_event_sec']}s")
                st.write(f"**Desat start:** {params['desat_start_sec']}s before end")
                st.write(f"**Desat end:** {params['desat_end_sec']}s after end")
            
            with col2:
                st.write(f"**Artifact filter:** {params['artifact_filter']}")
                st.write(f"**Desat threshold:** {params['desat_threshold']}%")
                st.write(f"**Global HB:** {'Enabled' if params['use_global_hb'] else 'Disabled'}")
            
            if params['use_global_hb']:
                baseline_text = 'Auto (95th percentile)' if params['preset_baseline'] == 0 else f'{params["preset_baseline"]:.1f}%'
                st.write(f"**Baseline SpO₂:** {baseline_text}")
        
        # Store in session state
        st.session_state.analysis_params = params
        
        # Warnings for non-default settings
        if preset_choice == "Custom":
            if params['pre_event_sec'] != 100:
                st.warning(f"⚠️ Pre-event window: {params['pre_event_sec']}s (Azarbarzin default: 100s)")
            if params['desat_start_sec'] != 60 or params['desat_end_sec'] != 120:
                st.warning(f"⚠️ Desat window: -{params['desat_start_sec']}s/+{params['desat_end_sec']}s (Azarbarzin: -60s/+120s)")
            if params['desat_threshold'] != 3:
                st.warning("⚠️ Using 4% threshold (non-AASM)")
            if params['artifact_filter'] != "Off":
                st.warning(f"⚠️ Artifact filtering enabled: {params['artifact_filter']}")
        elif preset_choice != "Azarbarzin 2019 (Default)":
            st.info(f"✨ Using **{preset_choice}** - results may differ from Azarbarzin 2019 baseline")
    
    # Extract params for use
    pre_event_sec = params['pre_event_sec']
    desat_start_sec = params['desat_start_sec']
    desat_end_sec = params['desat_end_sec']
    artifact_filter = params['artifact_filter']
    desat_threshold = params['desat_threshold']
    use_global_hb = params['use_global_hb']
    preset_baseline = params['preset_baseline']
    if desat_threshold == 4:
        st.warning("⚠️ Using 4% desaturation threshold (non-AASM standard)")
    if artifact_filter != "Off":
        st.info(f"ℹ️ Artifact filter enabled: {artifact_filter}")
    if not analyzer.flow_ch:
        st.warning("⚠️ No airflow channel found — AHI will be estimated from SpO₂ desaturations only")
    
    # --------------------------------------------------------------
    # ANALYSIS BUTTON
    # --------------------------------------------------------------
    st.markdown("---")
    
    if not st.session_state.analyzed:
        if st.button("🚀 Analyze File", type="primary", use_container_width=True):
            st.session_state.analyzed = True
            st.rerun()
    else:
        # Run analysis
        with st.spinner("🔬 Analyzing PSG data..."):
            results = analyzer.run_full_analysis(
                pre_event_sec=pre_event_sec,
                desat_start_sec=desat_start_sec,
                desat_end_sec=desat_end_sec,
                artifact_filter=artifact_filter,
                desat_threshold=desat_threshold,
                use_global_hb=use_global_hb,
                preset_baseline=preset_baseline,
                use_mit_st=st.session_state.use_mit_st
            )
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "AHI",
                f"{results['ahi']:.1f}",
                help="Apnea-Hypopnea Index (events per hour)"
            )
            if results.get('manual_ahi') is not None:
                delta = results['ahi'] - results['manual_ahi']
                st.caption(f"MIT Gold Std: {results['manual_ahi']:.1f} (Δ {delta:+.1f})")
        
        with col2:
            st.metric(
                f"ODI ({desat_threshold}%)",
                f"{results['odi']:.1f}",
                help=f"Oxygen Desaturation Index (≥{desat_threshold}% drops per hour)"
            )
        
        with col3:
            if len(results['events']) > 0:
                ci_str = f"[{results['ci'][0]:.1f}–{results['ci'][1]:.1f}]"
                st.metric(
                    "Obstructive HB",
                    f"{results['total_hb']:.1f}",
                    help=f"Event-specific Hypoxic Burden. 95% CI: {ci_str}"
                )
                st.caption(f"95% CI: {ci_str}")
            else:
                st.metric("Obstructive HB", "0.0")
        
        # Risk level
        risk_level = "Low"
        if results['total_hb'] >= 88:
            risk_level = "Very High"
            risk_color = "🔴"
        elif results['total_hb'] >= 53:
            risk_level = "High"
            risk_color = "🟠"
        elif results['total_hb'] >= 20:
            risk_level = "Moderate"
            risk_color = "🟡"
        else:
            risk_color = "🟢"
        
        st.markdown(f"### {risk_color} Risk Level: **{risk_level}**")
        
        # Global HB (if calculated)
        if results.get('global_hb') is not None:
            st.markdown("---")
            st.subheader("🌍 Global Hypoxic Burden")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Global HB",
                    f"{results['global_hb']:.2f} (%min)/h",
                    help="Total oxygen debt over entire sleep study"
                )
            with col2:
                st.metric(
                    "Baseline SpO₂",
                    f"{results['baseline_used']:.1f}%",
                    help="SpO₂ baseline used for calculation"
                )
        
        # Stage-specific results
        if results['stage_hb']:
            st.markdown("---")
            st.subheader("😴 Stage-Specific Metrics")
            
            stage_data = []
            for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                if stage in results['stage_hb']:
                    data = results['stage_hb'][stage]
                    stage_data.append({
                        'Stage': stage,
                        'Time (h)': f"{data['hrs']:.1f}",
                        'AHI': f"{data['AHI']:.1f}",
                        'ODI': f"{data['ODI']:.1f}",
                        'HB': f"{data['HB']:.2f}"
                    })
            
            if stage_data:
                import pandas as pd
                st.dataframe(pd.DataFrame(stage_data), use_container_width=True)
        
        # Report generation
        st.markdown("---")
        st.subheader("📄 Generate Report")
        
        col1, col2 = st.columns(2)
        with col1:
            proof_mode = st.selectbox(
                "Include proof plots",
                ["None", "Overlay (Azarbarzin-style)", "Full (all events)"],
                index=1
            )
        with col2:
            include_stages = st.checkbox("Include stage-specific results", value=True)
        
        if st.button("📥 Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_generator = PDFReportGenerator()
                buffer = pdf_generator.generate_report(
                    filename=edf_file.name,
                    results=results,
                    proof_mode=proof_mode,
                    include_stages=include_stages
                )
                
                st.download_button(
                    label="⬇️ Download Report",
                    data=buffer.getvalue(),
                    file_name=f"HB_Report_{edf_file.name.replace('.edf', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
        
        # Reset button
        if st.button("🔄 Analyze Another File", use_container_width=True):
            st.session_state.analyzed = False
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.rerun()

# =============================================
# BATCH ANALYSIS MODE
# =============================================
st.markdown("---")
st.markdown("### 📦 Batch Mode: Analyze Multiple Files")

batch_files = st.file_uploader(
    "Upload multiple PSG EDF files",
    type=["edf"],
    accept_multiple_files=True,
    key="batch_upload",
    help="Online: ≤5 files, ≤1 GB total. For larger batches, run locally."
)

if batch_files:
    n_files = len(batch_files)
    total_size_gb = sum(f.size for f in batch_files) / 1e9
    
    # Check limits
    if n_files > 5 or total_size_gb > 1.0:
        st.error("⚠️ **Batch Too Large for Online Use**")
        st.markdown(f"""
        Your batch has **{n_files} files** ({total_size_gb:.2f} GB total).
        
        **Online limits:**
        - ≤5 files
        - ≤1 GB total size
        
        **Solution:** Run the app locally (see instructions above) to process larger batches.
        """)
        st.stop()
    
    # Batch settings
    with st.expander("⚙️ Batch Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_include_stages = st.checkbox("Stage-specific results", value=True, key="batch_stages")
            batch_proof_mode = st.selectbox(
                "Proof plots",
                ["None", "Overlay (Azarbarzin-style)", "Full"],
                index=1,
                key="batch_proof"
            )
        
        with col2:
            batch_desat_threshold = st.selectbox(
                "Desaturation threshold",
                ["3%", "4%"],
                index=0,
                key="batch_desat"
            )
            batch_use_global_hb = st.checkbox(
                "Calculate Global HB",
                value=True,
                key="batch_global_hb",
                help="Calculate global hypoxic burden for each file"
            )
    
    # Initialize batch session state
    for key in ['batch_running', 'batch_paused', 'batch_progress', 'batch_results', 'batch_files_processed']:
        if key not in st.session_state:
            if 'progress' in key or 'processed' in key:
                st.session_state[key] = 0
            elif 'running' in key or 'paused' in key:
                st.session_state[key] = False
            else:
                st.session_state[key] = []
    
    # Batch control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_btn = st.button(
            "▶️ Run Batch",
            type="primary",
            disabled=st.session_state.batch_running,
            use_container_width=True
        )
    
    with col2:
        pause_btn = st.button(
            "⏸️ Pause",
            disabled=not st.session_state.batch_running or st.session_state.batch_paused,
            use_container_width=True
        )
    
    with col3:
        stop_btn = st.button(
            "⏹️ Stop",
            type="secondary",
            disabled=not st.session_state.batch_running,
            use_container_width=True
        )
    
    # Handle button clicks
    if stop_btn:
        for key in ['batch_running', 'batch_paused', 'batch_progress', 'batch_results', 'batch_files_processed']:
            if 'progress' in key or 'processed' in key:
                st.session_state[key] = 0
            elif 'running' in key or 'paused' in key:
                st.session_state[key] = False
            else:
                st.session_state[key] = []
        st.rerun()
    
    if pause_btn:
        st.session_state.batch_paused = True
        st.session_state.batch_running = False
        st.rerun()
    
    if st.session_state.batch_paused:
        if st.button("▶️ Resume Batch", type="primary", use_container_width=True):
            st.session_state.batch_running = True
            st.session_state.batch_paused = False
            st.rerun()
    
    # Progress indicators
    progress_bar = st.progress(st.session_state.batch_progress)
    status_text = st.empty()
    
    # Run batch processing
    if start_btn or (st.session_state.batch_running and not st.session_state.batch_paused):
        st.session_state.batch_running = True
        start_idx = st.session_state.batch_files_processed
        batch_summary_data = []
        
        desat_thresh_val = 3 if "3%" in batch_desat_threshold else 4
        
        for idx in range(start_idx, n_files):
            if not st.session_state.batch_running:
                break
            
            current_file = batch_files[idx]
            status_text.text(f"📂 Processing {current_file.name} ({idx+1}/{n_files})...")
            progress_bar.progress((idx + 0.1) / n_files)
            
            try:
                # Load file
                raw, temp_path = load_edf_file(current_file, f"temp_batch_{idx}.edf")
                
                if raw is None:
                    st.warning(f"⚠️ Skipping {current_file.name}: Could not load file")
                    continue
                
                # Run analysis
                analyzer = PSGAnalyzer(raw, temp_path)
                
                results = analyzer.run_full_analysis(
                    pre_event_sec=100,
                    desat_start_sec=60,
                    desat_end_sec=120,
                    artifact_filter="Off",
                    desat_threshold=desat_thresh_val,
                    use_global_hb=batch_use_global_hb,
                    preset_baseline=0.0,
                    use_mit_st=False
                )
                
                # Generate PDF
                pdf_generator = PDFReportGenerator()
                buffer = pdf_generator.generate_report(
                    filename=current_file.name,
                    results=results,
                    proof_mode=batch_proof_mode,
                    include_stages=batch_include_stages
                )
                
                # Store results
                st.session_state.batch_results.append((current_file.name, buffer))
                
                # Add to summary
                summary_row = {
                    'File': current_file.name,
                    'Duration (h)': f"{results['duration']:.1f}",
                    'AHI': f"{results['ahi']:.1f}",
                    'ODI': f"{results['odi']:.1f}",
                    'Obstructive HB': f"{results['total_hb']:.2f}"
                }
                
                if results.get('global_hb') is not None:
                    summary_row['Global HB'] = f"{results['global_hb']:.2f}"
                    summary_row['Baseline'] = f"{results['baseline_used']:.1f}%"
                
                batch_summary_data.append(summary_row)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing {current_file.name}: {str(e)}")
                continue
            
            # Update progress
            st.session_state.batch_files_processed = idx + 1
            progress_bar.progress((idx + 1) / n_files)
        
        # Generate master summary and ZIP
        status_text.text("📊 Generating master summary...")
        
        pdf_generator = PDFReportGenerator()
        master_buffer = pdf_generator.generate_batch_summary(batch_summary_data)
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add individual reports
            for filename, pdf_buffer in st.session_state.batch_results:
                zf.writestr(
                    f"Reports/HB_Report_{filename.replace('.edf', '')}.pdf",
                    pdf_buffer.getvalue()
                )
            
            # Add master summary
            zf.writestr("Master_Summary.pdf", master_buffer.getvalue())
        
        zip_buffer.seek(0)
        
        # Success message
        progress_bar.progress(1.0)
        status_text.text("✅ Batch processing complete!")
        st.success(f"**Batch Complete!** Generated {len(st.session_state.batch_results)} reports.")
        
        # Download button
        st.download_button(
            label=f"⬇️ Download All Reports ({len(st.session_state.batch_results)} files)",
            data=zip_buffer.getvalue(),
            file_name=f"HB_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        
        # Reset batch state
        for key in ['batch_running', 'batch_paused', 'batch_progress', 'batch_files_processed']:
            if 'progress' in key or 'processed' in key:
                st.session_state[key] = 0
            else:
                st.session_state[key] = False
        
        st.session_state.batch_results = []

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Hypoxic Burden Calculator</strong> — Open Source Sleep Apnea Analysis</p>
    <p>
        🐙 <a href="https://github.com/Apolloplectic/hypoxic-burden-edf">GitHub</a> • 
        📄 <a href="https://doi.org/10.5281/zenodo.17561726">DOI: 10.5281/zenodo.17561726</a> • 
        📧 <a href="mailto:sam.johnson9797@gmail.com">Contact</a>
    </p>
    <p><small>Built with Streamlit • MNE • {yasa_status} • WFDB</small></p>
    <p><small>Cite: Azarbarzin A, et al. <em>Eur Heart J</em> 2019;40:1149-1157</small></p>
</div>
""".format(yasa_status="YASA ✅" if YASA_AVAILABLE else "YASA ❌"), unsafe_allow_html=True)
