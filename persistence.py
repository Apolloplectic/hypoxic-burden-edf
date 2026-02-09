"""
Session Persistence for PSG Analysis
Saves analysis results to browser localStorage
"""

import streamlit as st
import json
from datetime import datetime


def save_analysis_to_session(filename, results, params):
    """
    Save analysis results to session state for persistence
    
    Parameters:
    -----------
    filename : str
        Name of analyzed file
    results : dict
        Analysis results
    params : dict
        Analysis parameters used
    """
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Create analysis record
    analysis_record = {
        'filename': filename,
        'timestamp': datetime.now().isoformat(),
        'ahi': float(results['ahi']),
        'odi': float(results['odi']),
        'total_hb': float(results['total_hb']),
        'duration': float(results['duration']),
        'risk_level': get_risk_level(results['total_hb']),
        'params': params
    }
    
    # Add global HB if available
    if results.get('global_hb'):
        analysis_record['global_hb'] = float(results['global_hb'])
        analysis_record['baseline'] = float(results['baseline_used'])
    
    # Add to history (keep last 10)
    st.session_state.analysis_history.insert(0, analysis_record)
    st.session_state.analysis_history = st.session_state.analysis_history[:10]


def get_risk_level(hb):
    """Get risk level from HB value"""
    if hb < 20:
        return "Low"
    elif hb < 53:
        return "Moderate"
    elif hb < 88:
        return "High"
    else:
        return "Very High"


def display_analysis_history():
    """
    Display recent analysis history in sidebar
    """
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Recent Analyses")
    
    for idx, record in enumerate(st.session_state.analysis_history):
        timestamp = datetime.fromisoformat(record['timestamp'])
        time_str = timestamp.strftime('%m/%d %H:%M')
        
        # Color code by risk
        risk_colors = {
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🟠",
            "Very High": "🔴"
        }
        risk_emoji = risk_colors.get(record['risk_level'], "⚪")
        
        with st.sidebar.expander(f"{risk_emoji} {record['filename'][:20]}... ({time_str})", expanded=False):
            st.write(f"**AHI:** {record['ahi']:.1f}")
            st.write(f"**HB:** {record['total_hb']:.1f}")
            st.write(f"**Risk:** {record['risk_level']}")
            st.write(f"**Duration:** {record['duration']:.1f}h")


def compare_with_previous():
    """
    Allow user to compare current analysis with previous ones
    
    Returns:
    --------
    dict or None
        Selected previous analysis for comparison
    """
    if 'analysis_history' not in st.session_state or len(st.session_state.analysis_history) < 2:
        return None
    
    st.markdown("### 🔄 Compare with Previous Analysis")
    
    # Create selection options
    options = ["None"] + [
        f"{rec['filename']} ({datetime.fromisoformat(rec['timestamp']).strftime('%m/%d %H:%M')})"
        for rec in st.session_state.analysis_history[1:]  # Skip current (first)
    ]
    
    selected = st.selectbox(
        "Select previous analysis to compare:",
        options,
        help="Compare current results with a previously analyzed file"
    )
    
    if selected == "None":
        return None
    
    # Find selected record
    selected_idx = options.index(selected)
    return st.session_state.analysis_history[selected_idx]


def display_comparison(current_results, previous_record):
    """
    Display side-by-side comparison of two analyses
    
    Parameters:
    -----------
    current_results : dict
        Current analysis results
    previous_record : dict
        Previous analysis record from history
    """
    st.markdown("#### Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Metric**")
        st.write("AHI")
        st.write("ODI")
        st.write("Hypoxic Burden")
        st.write("Risk Level")
    
    with col2:
        st.markdown("**Current**")
        st.write(f"{current_results['ahi']:.1f}")
        st.write(f"{current_results['odi']:.1f}")
        st.write(f"{current_results['total_hb']:.1f}")
        st.write(get_risk_level(current_results['total_hb']))
    
    with col3:
        st.markdown(f"**Previous** ({previous_record['filename'][:15]}...)")
        st.write(f"{previous_record['ahi']:.1f}")
        st.write(f"{previous_record['odi']:.1f}")
        st.write(f"{previous_record['total_hb']:.1f}")
        st.write(previous_record['risk_level'])
    
    # Calculate changes
    st.markdown("---")
    st.markdown("#### Changes")
    
    ahi_change = current_results['ahi'] - previous_record['ahi']
    hb_change = current_results['total_hb'] - previous_record['total_hb']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "AHI Change",
            f"{current_results['ahi']:.1f}",
            f"{ahi_change:+.1f}",
            delta_color="inverse"  # Lower is better
        )
    
    with col2:
        st.metric(
            "HB Change",
            f"{current_results['total_hb']:.1f}",
            f"{hb_change:+.1f}",
            delta_color="inverse"  # Lower is better
        )
