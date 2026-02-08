"""
Hypoxic Burden Calculator - Polysomnography Analysis Tool
Based on: Azarbarzin A, et al. European Heart Journal (2019)
DOI: 10.1093/eurheartj/ehy624

Author: Sam Johnson
Email: sam.johnson9797@gmail.com
GitHub: https://github.com/Apolloplectic/hypoxic-burden-edf
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io
import zipfile
import matplotlib.pyplot as plt
import json

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
# INITIALIZE SESSION STATE
# --------------------------------------------------------------
initialize_session_state()

# --------------------------------------------------------------
# SETTING PRESETS (Feature #11)
# --------------------------------------------------------------
PRESETS = {
    "Azarbarzin 2019 (Default)": {
        'pre_event_sec': 100,
        'desat_start_sec': 60,
        'desat_end_sec': 120,
        'desat_threshold': 3,
        'artifact_filter': 'Off',
        'use_global_hb': True,
        'preset_baseline': 0.0
    },
    "AASM 2023 Standard": {
        'pre_event_sec': 120,
        'desat_start_sec': 60,
        'desat_end_sec': 90,
        'desat_threshold': 3,
        'artifact_filter': 'Mild (10%/s)',
        'use_global_hb': True,
        'preset_baseline': 0.0
    },
    "Conservative (High Specificity)": {
        'pre_event_sec': 100,
        'desat_start_sec': 60,
        'desat_end_sec': 120,
        'desat_threshold': 4,
        'artifact_filter': 'Strict (5%/s)',
        'use_global_hb': False,
        'preset_baseline': 0.0
    },
    "Aggressive (High Sensitivity)": {
        'pre_event_sec': 80,
        'desat_start_sec': 45,
        'desat_end_sec': 150,
        'desat_threshold': 3,
        'artifact_filter': 'Off',
        'use_global_hb': True,
        'preset_baseline': 0.0
    },
    "Custom": None
}

# --------------------------------------------------------------
# FILE UPLOAD SECTION
# --------------------------------------------------------------
st.markdown("### 📁 Single File Analysis")
edf_file = st.file_uploader(
    "Upload PSG EDF file",
    type=["edf"],
    help="⚠️ Online version limited to 200 MB. For larger files (up to 2 GB), run locally.",
    key="single_file_upload"
)

# Show file info after upload
if edf_file is not None:
    file_size_mb = edf_file.size / 1e6
    
    if file_size_mb < 50:
        st.success(f"✅ **{edf_file.name}** uploaded ({file_size_mb:.1f} MB)")
    elif file_size_mb < 200:
        st.info(f"📁 **{edf_file.name}** uploaded ({file_size_mb:.1f} MB)")
    else:
        st.warning(f"⚠️ **{edf_file.name}** uploaded ({file_size_mb:.1f} MB) - Large file! For best performance, run locally.")

# Local run instructions expander
with st.expander("📦 File too large? Run locally (2 GB+ support) — no coding needed!", expanded=False):
    st.markdown("""
    ### **How to Run This App on Your Computer (2 GB+ Files)**
    **No programming experience required. Takes 5 minutes.**
    
    #### **Step 1: Install Python**
    - Download Python 3.9 or 3.10 from [python.org](https://python.org)
    - Run installer (check "Add to PATH")
    
    #### **Step 2: Download This App**
    ```bash
    git clone https://github.com/Apolloplectic/hypoxic-burden-edf.git
    cd hypoxic-burden-edf
    ```
    
    #### **Step 3: Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    
    #### **Step 4: Run With Large File Support**
    ```bash
    streamlit run app.py --server.maxUploadSize=4096
    ```
    
    The app will open in your browser automatically. Upload files up to **4 GB**!
    
    ---
    **Need help?** Email: `sam.johnson9797@gmail.com`
    """)

# --------------------------------------------------------------
# ANALYSIS SECTION
# --------------------------------------------------------------
if edf_file is not None:
    # Load EDF file
    with st.spinner(f"Loading {edf_file.name}..."):
        raw, temp_path = load_edf_file(edf_file)
    
    if raw is None:
        st.error("❌ Failed to load EDF file. Please check file format.")
        st.stop()
    
    st.success(f"✅ EDF loaded successfully! Duration: {raw.times[-1]/3600:.1f} hours")
    
    # Create analyzer
    analyzer = PSGAnalyzer(raw, temp_path)
    
    # Display channel detection
    st.markdown("#### 🔍 Detected Channels")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**SpO₂:** `{analyzer.spo2_ch or 'Not found'}`")
    with col2:
        st.write(f"**Airflow:** `{analyzer.flow_ch or 'Not found'}`")
    with col3:
        st.write(f"**EEG:** `{analyzer.eeg_ch or 'Not found'}`")
    
    if not analyzer.spo2_ch:
        st.error("❌ SpO₂ channel is required for analysis.")
        st.stop()
    
    # Check for MIT annotations
    use_mit_st = analyzer.manual_ahi is not None
    if use_mit_st:
        st.success("✅ MIT gold standard annotations found!")
    else:
        st.info("ℹ️ No MIT annotations found — using automated detection")
    
    # --------------------------------------------------------------
    # ADVANCED SETTINGS WITH PRESETS (Feature #11)
    # --------------------------------------------------------------
    with st.expander("⚙️ Advanced Settings", expanded=False):
        # Preset selector
        preset_choice = st.selectbox(
            "📋 Quick Presets",
            list(PRESETS.keys()),
            index=0,
            help="Select a preset configuration or choose 'Custom' for manual settings"
        )
        
        # Load preset or use custom
        if preset_choice != "Custom" and PRESETS[preset_choice]:
            params = PRESETS[preset_choice].copy()
            st.info(f"✨ Using **{preset_choice}** preset")
        else:
            params = PRESETS["Azarbarzin 2019 (Default)"].copy()
        
        st.markdown("#### Event Detection Parameters")
        col1, col2 = st.columns(2)
        with col1:
            if preset_choice == "Custom":
                params['pre_event_sec'] = st.selectbox(
                    "Pre-event baseline (s)", 
                    [60, 80, 100, 120], 
                    index=2
                )
                params['desat_start_sec'] = st.selectbox(
                    "Desat start before end (s)", 
                    [30, 45, 60, 90], 
                    index=2
                )
            else:
                st.write(f"**Pre-event baseline:** {params['pre_event_sec']}s")
                st.write(f"**Desat start:** {params['desat_start_sec']}s")
        
        with col2:
            if preset_choice == "Custom":
                params['desat_end_sec'] = st.selectbox(
                    "Desat end after end (s)", 
                    [90, 120, 150, 180], 
                    index=1
                )
                params['artifact_filter'] = st.selectbox(
                    "SpO₂ artifact filter", 
                    ["Off", "Mild (10%/s)", "Strict (5%/s)"], 
                    index=0
                )
            else:
                st.write(f"**Desat end:** {params['desat_end_sec']}s")
                st.write(f"**Artifact filter:** {params['artifact_filter']}")
        
        st.markdown("#### Scoring Parameters")
        col3, col4 = st.columns(2)
        with col3:
            if preset_choice == "Custom":
                scoring_rule = st.selectbox(
                    "Scoring Rule", 
                    ["3% (AASM)", "4% (Legacy)"], 
                    index=0
                )
                params['desat_threshold'] = 3 if "3%" in scoring_rule else 4
            else:
                st.write(f"**Desat threshold:** {params['desat_threshold']}%")
        
        with col4:
            if preset_choice == "Custom":
                params['use_global_hb'] = st.checkbox(
                    "Calculate Global Hypoxic Burden", 
                    value=True,
                    help="Whole-study desaturation burden"
                )
            else:
                st.write(f"**Global HB:** {'Enabled' if params['use_global_hb'] else 'Disabled'}")
        
        if params['use_global_hb']:
            if preset_choice == "Custom":
                params['preset_baseline'] = st.number_input(
                    "Baseline SpO₂ (%, 0=auto)",
                    min_value=0.0,
                    max_value=99.0,
                    value=0.0,
                    step=0.1,
                    help="0 = automatic 95th percentile"
                )
            else:
                baseline_text = 'Auto' if params['preset_baseline'] == 0 else f'{params["preset_baseline"]:.1f}%'
                st.write(f"**Baseline:** {baseline_text}")
                                                                                           
        # Warnings for non-default
        if preset_choice == "Azarbarzin 2019 (Default)":
            st.success("✅ Using validated Azarbarzin 2019 parameters")
        elif preset_choice != "Custom":
            st.warning(f"⚠️ Using {preset_choice} - results may differ from published values")

    # --------------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------------
    if not st.session_state.analyzed:
        if st.button("🚀 Analyze File", type="primary", use_container_width=True):
            st.session_state.analyzed = True
            st.rerun()
    else:
        # --------------------------------------------------------------
        # ANALYSIS WITH PROGRESS INDICATOR (Feature #2)
        # --------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔬 Analysis in Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1/7
        status_text.text("Step 1/7: Preprocessing SpO₂ signal...")
        progress_bar.progress(1/7)
        
        # Step 2/7
        status_text.text("Step 2/7: Detecting oxygen desaturation events...")
        progress_bar.progress(2/7)
        
        # Step 3/7
        status_text.text("Step 3/7: Detecting apnea/hypopnea events...")
        progress_bar.progress(3/7)
        
        # Step 4/7
        status_text.text("Step 4/7: Performing sleep staging...")
        progress_bar.progress(4/7)
        
        # Step 5/7  
        status_text.text("Step 5/7: Calculating hypoxic burden...")
        progress_bar.progress(5/7)
        
        # Run full analysis
        results = analyzer.run_full_analysis(
            pre_event_sec=params['pre_event_sec'],
            desat_start_sec=params['desat_start_sec'],
            desat_end_sec=params['desat_end_sec'],
            artifact_filter=params['artifact_filter'],
            desat_threshold=params['desat_threshold'],
            use_global_hb=params['use_global_hb'],
            preset_baseline=params['preset_baseline'],
            use_mit_st=use_mit_st
        )
        
        # Step 6/7
        status_text.text("Step 6/7: Computing confidence intervals...")
        progress_bar.progress(6/7)
        
        # Step 7/7
        status_text.text("Step 7/7: Finalizing results...")
        progress_bar.progress(1.0)
        
        status_text.text("✅ Analysis complete!")
        
        # Clear progress after 1 second
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # --------------------------------------------------------------
        # RESULTS DISPLAY
        # --------------------------------------------------------------
        st.markdown("---")
        
        # Calculate risk level
        total_hb = results['total_hb']
        if total_hb < 20:
            risk = "Low"
            risk_emoji = "🟢"
        elif total_hb < 53:
            risk = "Moderate"
            risk_emoji = "🟡"
        elif total_hb < 88:
            risk = "High"
            risk_emoji = "🟠"
        else:
            risk = "Very High"
            risk_emoji = "🔴"
        
        # Calculate severity
        ahi = results['ahi']
        if ahi < 5:
            severity = "Normal"
        elif ahi < 15:
            severity = "Mild OSA"
        elif ahi < 30:
            severity = "Moderate OSA"
        else:
            severity = "Severe OSA"
        
        # --------------------------------------------------------------
        # SUMMARY CARD AT TOP (Feature #7)
        # --------------------------------------------------------------
        st.markdown("## 📋 Executive Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown("#### 😴 Sleep Quality")
            st.write(f"**Duration:** {results['duration']:.1f} hours")
            staging_method = "YASA (AI)" if (YASA_AVAILABLE and analyzer.eeg_ch) else "Rule-based"
            if use_mit_st:
                staging_method = "MIT Gold Standard"
            st.write(f"**Staging:** {staging_method}")
            
        with summary_col2:
            st.markdown("#### 🫁 Breathing Events")
            st.write(f"**AHI:** {ahi:.1f} events/hr")
            st.write(f"**Severity:** {severity}")
            st.write(f"**ODI:** {results['odi']:.1f}")
            
        with summary_col3:
            st.markdown("#### ⚕️ Risk Assessment")
            st.write(f"**Hypoxic Burden:** {total_hb:.1f}")
            st.write(f"**Risk Level:** {risk_emoji} {risk}")
        
        st.markdown("---")
        
        # --------------------------------------------------------------
        # PRIMARY METRICS
        # --------------------------------------------------------------
        st.markdown("## 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "**AHI**",
                f"{ahi:.1f}",
                help="Apnea-Hypopnea Index: breathing pauses per hour. <5=normal, 5-15=mild, 15-30=moderate, >30=severe"
            )
        
        with col2:
            st.metric(
                f"**ODI ({params['desat_threshold']}%)**",
                f"{results['odi']:.1f}",
                help=f"Oxygen Desaturation Index: ≥{params['desat_threshold']}% drops per hour"
            )
        
        with col3:
            ci_text = f"[{results['ci'][0]:.1f}–{results['ci'][1]:.1f}]"
            st.metric(
                "**Obstructive HB**",
                f"{total_hb:.1f}",
                delta=f"95% CI: {ci_text}",
                delta_color="off",
                help="Event-specific hypoxic burden (Azarbarzin method)"
            )
        
        # --------------------------------------------------------------
        # RISK VISUALIZATION (Feature #4)
        # --------------------------------------------------------------
        st.markdown(f"### {risk_emoji} Risk Level: {risk}")
        
        # Risk scale with color-coded progress bar
        risk_percentage = min(total_hb / 88, 1.0)
        
        if risk == "Low":
            st.progress(risk_percentage, text=f"Low Risk ({total_hb:.1f} / 88)")
        elif risk == "Moderate":
            st.progress(risk_percentage, text=f"Moderate Risk ({total_hb:.1f} / 88)")
        elif risk == "High":
            st.progress(risk_percentage, text=f"High Risk ({total_hb:.1f} / 88)")
        else:
            st.progress(1.0, text=f"Very High Risk ({total_hb:.1f}+)")
        
        st.caption("Risk thresholds: Low <20 | Moderate 20-53 | High 53-88 | Very High ≥88")
        
        # --------------------------------------------------------------
        # COLLAPSIBLE SECTIONS (Feature #3)
        # --------------------------------------------------------------
        
        # Global Hypoxic Burden
        if results.get('global_hb') is not None:
            with st.expander("🌍 Global Hypoxic Burden Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Global HB",
                        f"{results['global_hb']:.1f} (%min)/h",
                        help="Total desaturation area over entire sleep study"
                    )
                with col2:
                    st.metric(
                        "Baseline SpO₂",
                        f"{results['baseline_used']:.1f}%",
                        help="Automatically calculated as 95th percentile" if params['preset_baseline'] == 0 else "User-specified baseline"
                    )
                
                st.markdown("**Methodology:**")
                st.write("Global HB measures the total 'oxygen debt' across the entire sleep study, "
                        "not just during apnea events. Calculated as the area between the SpO₂ curve "
                        "and the baseline, normalized per hour.")
        
        # Sleep Architecture with Hypnogram (Features #3, #12)
        with st.expander("😴 Sleep Architecture & Hypnogram", expanded=True):
            if analyzer.stages and len(analyzer.stages) > 0:
                # Sleep Hypnogram (Feature #12)
                st.markdown("#### Sleep Hypnogram")
                
                stage_map = {'W': 0, 'REM': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'Unknown': 2}
                stage_values = [stage_map.get(s, 2) for s in analyzer.stages]
                time_epochs = np.arange(len(stage_values)) * 0.5  # 30s = 0.5 min
                
                fig_hypno, ax = plt.subplots(figsize=(12, 3))
                ax.plot(time_epochs, stage_values, linewidth=1.0, color='#1f77b4')
                ax.fill_between(time_epochs, stage_values, 0, alpha=0.3, color='#1f77b4')
                ax.set_yticks([0, 1, 2, 3, 4])
                ax.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Sleep Stage')
                ax.set_title('Sleep Architecture Over Time')
                ax.grid(True, alpha=0.2, axis='x')
                ax.set_ylim(-0.5, 4.5)
                
                st.pyplot(fig_hypno)
                plt.close(fig_hypno)
                
                st.markdown("---")
                
                # Stage-Specific Table
                st.markdown("#### Stage-Specific Metrics")
                
                if results['stage_hb']:
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
                        df_stages = pd.DataFrame(stage_data)
                        st.dataframe(df_stages, use_container_width=True, hide_index=True)
                        
                        # Stage Comparison Charts (Feature #6)
                        st.markdown("#### Stage Comparison Charts")
                        
                        stages = [d['Stage'] for d in stage_data]
                        ahis = [float(d['AHI']) for d in stage_data]
                        hbs = [float(d['HB']) for d in stage_data]
                        
                        fig_charts, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
                        
                        # AHI by stage
                        bars1 = ax1.bar(stages, ahis, color=colors)
                        ax1.set_ylabel('AHI (events/hour)', fontsize=11)
                        ax1.set_xlabel('Sleep Stage', fontsize=11)
                        ax1.set_title('AHI by Sleep Stage', fontsize=12, fontweight='bold')
                        ax1.grid(axis='y', alpha=0.3, linestyle='--')
                        ax1.set_ylim(0, max(ahis) * 1.2 if max(ahis) > 0 else 10)
                        
                        # Add value labels on bars
                        for bar, val in zip(bars1, ahis):
                            height = bar.get_height()
                            if height > 0:
                                ax1.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{val:.1f}',
                                        ha='center', va='bottom', fontsize=9)
                        
                        # HB by stage
                        bars2 = ax2.bar(stages, hbs, color=colors)
                        ax2.set_ylabel('Hypoxic Burden (%min/h)', fontsize=11)
                        ax2.set_xlabel('Sleep Stage', fontsize=11)
                        ax2.set_title('Hypoxic Burden by Sleep Stage', fontsize=12, fontweight='bold')
                        ax2.grid(axis='y', alpha=0.3, linestyle='--')
                        ax2.set_ylim(0, max(hbs) * 1.2 if max(hbs) > 0 else 10)
                        
                        # Add value labels on bars
                        for bar, val in zip(bars2, hbs):
                            height = bar.get_height()
                            if height > 0:
                                ax2.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{val:.1f}',
                                        ha='center', va='bottom', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_charts)
                        plt.close(fig_charts)
                    else:
                        st.warning("⚠️ No sleep stages detected in this recording")
                else:
                    st.warning("⚠️ Sleep staging failed - no stage-specific results available")
            else:
                st.info("ℹ️ Sleep staging not available for this file")
        
        # MIT Comparison (if available)
        if results.get('manual_ahi') is not None:
            with st.expander("🆚 Algorithm vs Manual Scoring", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("App AHI", f"{results['ahi']:.1f}")
                with col2:
                    st.metric("Manual (MIT) AHI", f"{results['manual_ahi']:.1f}")
                with col3:
                    delta = results['ahi'] - results['manual_ahi']
                    st.metric("Difference", f"{delta:+.1f}", delta_color="off")
                
                # Show accuracy
                if results['manual_ahi'] > 0:
                    percent_diff = abs(delta / results['manual_ahi'] * 100)
                    accuracy = max(0, 100 - percent_diff)
                    st.progress(accuracy / 100, text=f"Agreement: {accuracy:.1f}%")
        
        # Advanced Metrics
        with st.expander("📈 Advanced Metrics & Methodology", expanded=False):
            st.markdown("#### Calculation Parameters")
            st.write(f"**Pre-event baseline window:** {params['pre_event_sec']} seconds")
            st.write(f"**Desaturation window:** -{params['desat_start_sec']}s to +{params['desat_end_sec']}s")
            st.write(f"**Desaturation threshold:** {params['desat_threshold']}%")
            st.write(f"**Artifact filter:** {params['artifact_filter']}")
            
            st.markdown("#### Methodology")
            st.write("**Obstructive HB:** Event-specific method from Azarbarzin et al. "
                    "Calculates area under desaturation curve for each apnea/hypopnea event.")
            st.write("**Global HB:** Total desaturation area below baseline over entire sleep study.")
            st.write("**Confidence Intervals:** Bootstrap resampling with 1000 iterations.")
            
            if analyzer.stages:
                staging_method = "MIT gold standard" if use_mit_st else \
                                "YASA deep learning" if (YASA_AVAILABLE and analyzer.eeg_ch) else \
                                "Rule-based spectral analysis"
                st.write(f"**Sleep Staging:** {staging_method}")
        
        st.markdown("---")
        
        # --------------------------------------------------------------
        # EXPORT OPTIONS (Feature #10)
        # --------------------------------------------------------------
        st.markdown("### 📤 Export & Download Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # PDF Report
            st.markdown("#### 📄 PDF Report")
            
            col_pdf1, col_pdf2 = st.columns(2)
            with col_pdf1:
                proof_mode = st.selectbox(
                    "Proof plots",
                    ["None", "Overlay (Azarbarzin-style)", "Full (all events)"],
                    index=1,
                    key="proof_select"
                )
            with col_pdf2:
                include_stages = st.checkbox("Include stages", value=True, key="stages_check")
            
            if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    pdf_generator = PDFReportGenerator()
                    buffer = pdf_generator.generate_report(
                        filename=edf_file.name,
                        results=results,
                        proof_mode=proof_mode,
                        include_stages=include_stages
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=buffer.getvalue(),
                        file_name=f"HB_Report_{edf_file.name.replace('.edf', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
        
        with export_col2:
            # CSV Export
            st.markdown("#### 📊 Data Export")
            
            if results['stage_hb']:
                # Prepare CSV data
                csv_data = []
                for stage, data in results['stage_hb'].items():
                    csv_data.append({
                        'Stage': stage,
                        'Duration_hours': data['hrs'],
                        'AHI': data['AHI'],
                        'ODI': data['ODI'],
                        'Hypoxic_Burden': data['HB']
                    })
                
                df_csv = pd.DataFrame(csv_data)
                csv_string = df_csv.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Stage Data (CSV)",
                    data=csv_string,
                    file_name=f"stage_data_{edf_file.name.replace('.edf', '')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No stage data available")
        
        with export_col3:
            # JSON Export
            st.markdown("#### 🔬 Raw Data")
            
            # Prepare JSON (convert numpy types to native Python)
            json_results = {
                'filename': edf_file.name,
                'analysis_date': datetime.now().isoformat(),
                'duration_hours': float(results['duration']),
                'ahi': float(results['ahi']),
                'odi': float(results['odi']),
                'obstructive_hb': float(results['total_hb']),
                'ci_lower': float(results['ci'][0]),
                'ci_upper': float(results['ci'][1]),
                'risk_level': risk,
                'stage_specific': {k: {kk: float(vv) for kk, vv in v.items()} 
                                  for k, v in results['stage_hb'].items()} if results['stage_hb'] else {}
            }
            
            if results.get('global_hb') is not None:
                json_results['global_hb'] = float(results['global_hb'])
                json_results['baseline_spo2'] = float(results['baseline_used'])
            
            json_string = json.dumps(json_results, indent=2)
            
            st.download_button(
                label="🔬 Download Raw Data (JSON)",
                data=json_string,
                file_name=f"analysis_{edf_file.name.replace('.edf', '')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Analyze Another File", use_container_width=True):
            st.session_state.analyzed = False
            st.rerun()

# --------------------------------------------------------------
# BATCH MODE
# --------------------------------------------------------------
st.markdown("---")
st.markdown("### 📦 Batch Mode: Analyze Multiple Files")

st.info("📝 **Note:** Batch mode uses the same analysis pipeline as single file mode. "
        "For >5 files or >1 GB total, run locally for best performance.")

batch_files = st.file_uploader(
    "Upload multiple PSG EDF files",
    type=["edf"],
    accept_multiple_files=True,
    key="batch_upload",
    help="⚠️ Online: ≤5 files, ≤1 GB total. For larger batches, run locally."
)

if batch_files:
    n_files = len(batch_files)
    total_size_gb = sum(f.size for f in batch_files) / 1e9
    
    # Size check
    if n_files > 5 or total_size_gb > 1.0:
        st.error("**⚠️ BATCH TOO LARGE FOR ONLINE USE**")
        st.markdown("""
        **This batch requires local deployment:**
        - **>5 files** or **>1 GB** → too slow for cloud
        - **Solution**: Run locally (supports up to 50 files, 4 GB total)
        
        See "File too large?" section above for local setup instructions.
        """)
        st.stop()
    
    st.success(f"✅ {n_files} files uploaded ({total_size_gb:.2f} GB total)")
    
    # Batch settings
    with st.expander("📋 Batch Report Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_proof_mode = st.selectbox(
                "Proof plots",
                ["None", "Overlay (Azarbarzin-style)", "Full (all events)"],
                index=1,
                key="batch_proof"
            )
            batch_include_stages = st.checkbox(
                "Include stage-specific results",
                value=True,
                key="batch_stages"
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
                key="batch_global_hb"
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
    
    # Handle stop/pause
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
    progress_bar = st.progress(st.session_state.batch_progress / n_files if n_files > 0 else 0)
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

            # BATCH MODE ERROR HANDLING PATCH
# Add this improved error handling to your batch processing loop

# Replace the try/except block in batch processing (around line 180-220) with this:

try:
    # Load file
    raw, temp_path = load_edf_file(current_file, f"temp_batch_{idx}.edf")
    
    if raw is None:
        st.warning(f"⚠️ Skipping {current_file.name}: Could not load file")
        continue
    
    # Validate file before analysis
    # Check that all channels have data
    if raw.times[-1] == 0:
        st.warning(f"⚠️ Skipping {current_file.name}: File has no duration")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        continue
    
    # Run analysis with additional error handling
    analyzer = PSGAnalyzer(raw, temp_path)
    
    # Check for SpO2 channel
    if not analyzer.spo2_ch:
        st.warning(f"⚠️ Skipping {current_file.name}: No SpO₂ channel found")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        continue
    
    # Run analysis
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
    
    st.success(f"✅ Completed: {current_file.name}")
    
except ValueError as e:
    if "All arrays must be of the same length" in str(e):
        st.error(f"❌ {current_file.name}: Mismatched channel lengths (corrupted file)")
    else:
        st.error(f"❌ {current_file.name}: {str(e)}")
    
    # Cleanup on error
    if 'temp_path' in locals() and os.path.exists(temp_path):
        os.remove(temp_path)
    continue

except Exception as e:
    st.error(f"❌ Error processing {current_file.name}: {str(e)}")
    
    # Cleanup on error
    if 'temp_path' in locals() and os.path.exists(temp_path):
        os.remove(temp_path)
    continue
            
            # Update progress
            st.session_state.batch_files_processed = idx + 1
            st.session_state.batch_progress = idx + 1
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
        
        # Download ZIP
        st.download_button(
            label="📥 Download All Reports (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"HB_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        
        # Reset batch state
        if st.button("🔄 Process Another Batch", use_container_width=True):
            for key in ['batch_running', 'batch_paused', 'batch_progress', 'batch_files_processed']:
                if 'progress' in key or 'processed' in key:
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = False
            st.session_state.batch_results = []
            st.rerun()

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.markdown("---")
st.markdown("### 📚 About & Resources")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Citation:**")
    st.caption("Azarbarzin A, et al. *Eur Heart J* 2019;40:1149-1157")

with footer_col2:
    st.markdown("**Status:**")
    yasa_status = "✅ YASA Available" if YASA_AVAILABLE else "ℹ️ Rule-based staging"
    st.caption(yasa_status)

with footer_col3:
    st.markdown("**Links:**")
    st.caption("[GitHub](https://github.com/Apolloplectic/hypoxic-burden-edf) | "
              "[DOI: 10.5281/zenodo.17561726](https://doi.org/10.5281/zenodo.17561726)")

st.caption("Built with Streamlit + MNE + YASA + WFDB")
