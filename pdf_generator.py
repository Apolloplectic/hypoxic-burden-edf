"""
PDF Report Generator for Hypoxic Burden Calculator
"""

import io
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import pandas as pd

from config import RISK_THRESHOLDS


class PDFReportGenerator:
    """Generate PDF reports for hypoxic burden analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=20
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
    
    def generate_report(self, filename, results, proof_mode="Overlay", include_stages=True):
        """
        Generate complete PDF report
        
        Parameters:
        -----------
        filename : str
            Original EDF filename
        results : dict
            Analysis results
        proof_mode : str
            "None", "Overlay", or "Full"
        include_stages : bool
            Include stage-specific results
        
        Returns:
        --------
        io.BytesIO
            PDF buffer
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Hypoxic Burden Analysis Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # File info
        story.append(Paragraph(f"<b>File:</b> {filename}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Study Duration:</b> {results['duration']:.2f} hours", self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Main metrics
        story.append(Paragraph("Primary Metrics", self.heading_style))
        
        metrics_data = [
            ["Metric", "Value", "Interpretation"],
            ["AHI (events/hour)", f"{results['ahi']:.1f}", self._interpret_ahi(results['ahi'])],
            [f"ODI (\u22653% drops/hour)", f"{results['odi']:.1f}", ""],
            [
                "Hypoxic Burden (Parekh)" if results.get('method') == 'parekh' else "Obstructive Hypoxic Burden",
                f"{results['total_hb']:.2f} (%min)/h",
                self._interpret_hb(results['total_hb'])
            ],
        ]
        
        if len(results['events']) > 0:
            ci_low, ci_high = results['ci']
            metrics_data.append(["95% Confidence Interval", f"[{ci_low:.2f} – {ci_high:.2f}]", ""])
        
        # Manual AHI comparison
        if results.get('manual_ahi') is not None:
            delta = results['ahi'] - results['manual_ahi']
            metrics_data.append([
                "MIT Gold Standard AHI",
                f"{results['manual_ahi']:.1f}",
                f"Δ = {delta:+.1f}"
            ])
        
        # Global HB
        if results.get('global_hb') is not None:
            bl_method = results.get('baseline_method', 'stage_specific')
            if bl_method == 'floating':
                window_min = results.get('floating_window_sec', 600) // 60
                baseline_note = f"Floating baseline ({window_min}-min window)"
            elif bl_method == 'stage_specific':
                baseline_note = f"Wt. Avg Baseline: {results['baseline_used']:.1f}% (stage-specific)"
            else:
                baseline_note = f"Baseline: {results['baseline_used']:.1f}% (whole-night)"
            metrics_data.append([
                "Global Hypoxic Burden",
                f"{results['global_hb']:.2f} (%min)/h",
                baseline_note
            ])
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ])
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk assessment
        risk_level, risk_color = self._get_risk_level(results['total_hb'])
        story.append(Paragraph(f"<b>Risk Stratification:</b> {risk_level}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Stage-specific results
        if include_stages and results['stage_hb']:
            story.append(Paragraph("Stage-Specific Metrics", self.heading_style))
            
            stage_data = [["Sleep Stage", "Time (hours)", "AHI", "ODI", "HB (%min)/h"]]
            for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                if stage in results['stage_hb']:
                    data = results['stage_hb'][stage]
                    stage_data.append([
                        self._format_stage_name(stage),
                        f"{data['hrs']:.1f}",
                        f"{data['AHI']:.1f}",
                        f"{data['ODI']:.1f}",
                        f"{data['HB']:.2f}"
                    ])
            
            stage_table = Table(stage_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1*inch, 1.2*inch])
            stage_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ])
            
            story.append(stage_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Proof plots
        if proof_mode != "None" and results['events']:
            story.append(PageBreak())
            story.append(Paragraph("Desaturation Event Analysis", self.heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            if proof_mode == "Overlay":
                fig = self._plot_overlay_events(results['events'])
                img = self._fig_to_image(fig, width=6*inch, height=3*inch)
                story.append(img)
                plt.close(fig)
                
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(
                    f"<i>Ensemble average of {len(results['events'])} desaturation events. "
                    "Individual events shown in gray, mean in blue.</i>",
                    self.styles['Normal']
                ))
            
            elif proof_mode == "Full":
                # Show first 10 events
                n_show = min(10, len(results['events']))
                story.append(Paragraph(
                    f"<i>Showing first {n_show} of {len(results['events'])} events</i>",
                    self.styles['Normal']
                ))
                story.append(Spacer(1, 0.2*inch))
                
                for i in range(n_show):
                    ev = results['events'][i]
                    fig = self._plot_single_event(ev, i+1)
                    img = self._fig_to_image(fig, width=5*inch, height=2.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.15*inch))
                    plt.close(fig)
                    
                    if (i + 1) % 3 == 0 and i < n_show - 1:
                        story.append(PageBreak())
        
        # Methodology
        story.append(PageBreak())
        story.append(Paragraph("Methodology", self.heading_style))

        if results.get('method') == 'parekh':
            story.append(Paragraph(
                "<b>Hypoxic Burden (Parekh 2023):</b> Automated SpO2 nadir detection method from "
                "Parekh et al., AJRCCM 2023;208(11):1216-1226. Desaturation events identified "
                "by peak prominence (>=3%) in the SpO2 signal without requiring manually scored "
                "respiratory events. Each event's baseline defined by flanking left and right SpO2 "
                "peaks, with area calculated between the peak-connecting line and the SpO2 trace.",
                self.styles['Normal']
            ))
        else:
            story.append(Paragraph(
                "<b>Obstructive Hypoxic Burden:</b> Event-specific method from Azarbarzin et al., "
                "European Heart Journal 2019;40:1149-1157. Calculates area under desaturation "
                "curve for each apnea/hypopnea event.",
                self.styles['Normal']
            ))
        story.append(Spacer(1, 0.1*inch))
        
        if results.get('global_hb') is not None:
            bl_method = results.get('baseline_method', 'stage_specific')

            if bl_method == 'floating':
                window_min = results.get('floating_window_sec', 600) // 60
                story.append(Paragraph(
                    f"<b>Global Hypoxic Burden:</b> Total desaturation area below a floating "
                    f"baseline over entire sleep study. The floating baseline uses a trailing "
                    f"{window_min}-minute window: bottom 25% of SpO2 values are removed, then "
                    f"the 95th percentile of the remaining values is used. This adapts to local "
                    f"SpO2 conditions without requiring sleep staging data. "
                    f"Mean baseline: {results['baseline_used']:.1f}%.",
                    self.styles['Normal']
                ))
            elif bl_method == 'stage_specific':
                stage_baselines = results.get('stage_baselines', {})
                if stage_baselines:
                    bl_details = ", ".join(
                        f"{s}: {stage_baselines[s]:.1f}%"
                        for s in ['W', 'N1', 'N2', 'N3', 'REM']
                        if s in stage_baselines
                    )
                    story.append(Paragraph(
                        "<b>Global Hypoxic Burden:</b> Total desaturation area below stage-specific "
                        "baselines over entire sleep study. Baselines derived from 95th percentile of "
                        "SpO2 in event-free 30s epochs per sleep stage, avoiding contamination by "
                        f"respiratory event desaturations. Weighted avg baseline: "
                        f"{results['baseline_used']:.1f}%. Stage baselines: {bl_details}.",
                        self.styles['Normal']
                    ))
                else:
                    story.append(Paragraph(
                        "<b>Global Hypoxic Burden:</b> Total desaturation area below baseline over entire "
                        f"sleep study. Baseline automatically calculated as 95th percentile of SpO2 "
                        f"({results['baseline_used']:.1f}%) to exclude desaturation outliers.",
                        self.styles['Normal']
                    ))
            else:  # whole_night
                story.append(Paragraph(
                    "<b>Global Hypoxic Burden:</b> Total desaturation area below baseline over entire "
                    f"sleep study. Whole-night baseline calculated as 95th percentile of SpO2 "
                    f"({results['baseline_used']:.1f}%).",
                    self.styles['Normal']
                ))
            story.append(Spacer(1, 0.1*inch))
        
        if results.get('use_mit_st'):
            story.append(Paragraph(
                "<b>Sleep Staging:</b> MIT-annotated gold standard from SHHS/slpdb database.",
                self.styles['Normal']
            ))
        else:
            from config import YASA_AVAILABLE
            if YASA_AVAILABLE:
                story.append(Paragraph(
                    "<b>Sleep Staging:</b> Automated using YASA deep learning model.",
                    self.styles['Normal']
                ))
            else:
                story.append(Paragraph(
                    "<b>Sleep Staging:</b> Rule-based spectral analysis of EEG.",
                    self.styles['Normal']
                ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    def generate_batch_summary(self, batch_data):
        """
        Generate master summary PDF for batch analysis
        
        Parameters:
        -----------
        batch_data : list of dict
            Summary data for each file
        
        Returns:
        --------
        io.BytesIO
            PDF buffer
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch)
        
        story = []
        
        story.append(Paragraph("Batch Analysis Summary", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"<b>Files Analyzed:</b> {len(batch_data)}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Report Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Create summary table
        headers = list(batch_data[0].keys())
        table_data = [headers]
        
        for row in batch_data:
            table_data.append([row[h] for h in headers])
        
        # Calculate column widths dynamically
        n_cols = len(headers)
        col_width = 6.5 * inch / n_cols
        
        summary_table = Table(table_data, colWidths=[col_width] * n_cols)
        summary_table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ])
        
        story.append(summary_table)
        
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    def _plot_overlay_events(self, events):
        """Create ensemble-average overlay plot (Azarbarzin style)"""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        t_grid = np.linspace(-60, 120, 181)
        all_spo2 = []
        
        for ev in events:
            t_raw = ev['win_df']['time'].values - ev['end_t']
            s_raw = ev['win_df']['spo2'].values
            
            # Interpolate to common grid
            s_interp = np.interp(t_grid, t_raw, s_raw, left=np.nan, right=np.nan)
            ax.plot(t_grid, s_interp, color='lightgray', alpha=0.3, linewidth=0.8)
            all_spo2.append(s_interp)
        
        # Calculate mean
        all_spo2 = np.array(all_spo2)
        mean_spo2 = np.nanmean(all_spo2, axis=0)
        
        ax.plot(t_grid, mean_spo2, color='darkblue', linewidth=3, label=f'Mean (n={len(events)})')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Event end')
        
        ax.set_title("Ensemble-Average Desaturation Curve", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time relative to event end (seconds)", fontsize=11)
        ax.set_ylabel("SpO₂ (%)", fontsize=11)
        ax.set_ylim(75, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_single_event(self, ev, event_num):
        """Plot a single desaturation event"""
        fig, ax = plt.subplots(figsize=(7, 3))
        
        t = ev['win_df']['time'].values - ev['end_t']
        s = ev['win_df']['spo2'].values
        
        ax.plot(t, s, 'red', linewidth=2)
        ax.axhline(ev['baseline'], color='green', linestyle='--', label=f"Baseline ({ev['baseline']:.1f}%)")
        ax.fill_between(t, s, ev['baseline'], where=(s < ev['baseline']), color='red', alpha=0.3)
        
        ax.set_title(f"Event #{event_num} at {ev['end_t']:.0f}s (HB contribution: {ev['hb_contrib']:.2f})", fontsize=12)
        ax.set_xlabel("Time from event end (s)", fontsize=10)
        ax.set_ylabel("SpO₂ (%)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _fig_to_image(self, fig, width=5*inch, height=3*inch):
        """Convert matplotlib figure to ReportLab Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        return Image(buf, width=width, height=height)
    
    def _interpret_ahi(self, ahi):
        """Interpret AHI severity"""
        if ahi < 5:
            return "Normal"
        elif ahi < 15:
            return "Mild OSA"
        elif ahi < 30:
            return "Moderate OSA"
        else:
            return "Severe OSA"
    
    def _interpret_hb(self, hb):
        """Interpret HB risk level"""
        if hb < RISK_THRESHOLDS['moderate']:
            return "Low risk"
        elif hb < RISK_THRESHOLDS['high']:
            return "Moderate risk"
        elif hb < RISK_THRESHOLDS['very_high']:
            return "High risk"
        else:
            return "Very high risk"

    def _get_risk_level(self, hb):
        """Get risk level and color"""
        if hb < RISK_THRESHOLDS['moderate']:
            return "Low", colors.green
        elif hb < RISK_THRESHOLDS['high']:
            return "Moderate", colors.yellow
        elif hb < RISK_THRESHOLDS['very_high']:
            return "High", colors.orange
        else:
            return "Very High", colors.red
    
    def _format_stage_name(self, stage):
        """Format sleep stage name for display"""
        names = {
            'W': 'Wake',
            'N1': 'Stage N1',
            'N2': 'Stage N2',
            'N3': 'Stage N3',
            'REM': 'REM Sleep'
        }
        return names.get(stage, stage)
