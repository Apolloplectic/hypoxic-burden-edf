# Clinical Validation Module - Documentation

## 🎯 Purpose

The Clinical Validation Module allows you to compare automated PSG analysis results with manual/clinical scoring to:
- Validate algorithm accuracy
- Quality assurance checks
- Research validation for publications
- Training and learning

---

## 📍 Location in App

**Where:** After "Stage-Specific Metrics", before "Interactive SpO₂ Analysis"  
**Format:** Collapsible expander (collapsed by default)  
**Label:** "🏥 Clinical Validation Mode"

---

## 🔧 How to Use

### Step 1: Run Normal Analysis
1. Upload your EDF file
2. Configure analysis parameters
3. Click "Analyze"
4. Review automated results

### Step 2: Enter Manual Data
1. Expand "🏥 Clinical Validation Mode"
2. Enter metrics from clinical report:
   - **Manual AHI** (required for comparison)
   - **Manual ODI** (optional)
   - **Total Events** (optional)
   - **Total Sleep Time** (optional)
   - **Baseline SpO₂** (optional)

### Step 3: Review Comparison
Once you enter at least the Manual AHI, the comparison automatically appears showing:
- Side-by-side metrics table
- Differences (absolute and percentage)
- Agreement statistics
- Overall quality assessment

### Step 4: Export Results (Optional)
Click "Export Validation Report" to download a JSON file with all comparison data.

---

## 📊 Metrics Compared

| Metric | Description | Source |
|--------|-------------|--------|
| AHI | Apnea-Hypopnea Index | Events per hour of sleep |
| ODI | Oxygen Desaturation Index | Desaturations per hour |
| Total Events | Count of apnea/hypopnea events | Event detector |
| TST | Total Sleep Time | Hours of sleep |
| Baseline SpO₂ | Baseline oxygen saturation | SpO₂ analysis |

---

## 📈 Agreement Statistics

### Mean Absolute Percentage Error (MAPE)
Average percentage difference across all entered metrics.

**Interpretation:**
- **< 10%** 🟢 Good agreement
- **10-20%** 🟡 Fair agreement  
- **> 20%** 🔴 Poor agreement

### Metrics Within 10%
Count of how many metrics differ by less than 10%.

**Example:**
- 4/5 metrics within 10% = Good overall agreement
- 2/5 metrics within 10% = Review recommended

---

## 🔬 Use Cases

### 1. Algorithm Validation (Research)
**Scenario:** Publishing research using this tool

**Workflow:**
1. Collect 10-20 PSGs with clinical reports
2. Run all through your app
3. Enter manual values for each
4. Export validation reports
5. Calculate aggregate statistics:
   - Mean MAPE across all files
   - Correlation coefficient
   - Bland-Altman analysis

**Include in Methods Section:**
> "Automated analysis was validated against manual scoring from [N] PSG studies. 
> Mean absolute percentage error was [X]%, with [Y]% of metrics showing <10% difference."

---

### 2. Quality Assurance (Clinical)
**Scenario:** Spot-checking automated analysis

**Workflow:**
1. Run analysis on patient file
2. Compare key metrics (AHI, ODI) with clinical report
3. If MAPE > 15%, flag for manual review
4. Investigate discrepancies

**Decision Rule:**
- MAPE < 10%: Accept automated results
- MAPE 10-20%: Review but likely acceptable
- MAPE > 20%: Manual review required

---

### 3. Method Development (Research)
**Scenario:** Testing new baseline calculation

**Workflow:**
1. Implement new method
2. Run on test set
3. Compare with gold standard
4. Choose method with lowest MAPE

---

### 4. Training (Education)
**Scenario:** Learning sleep medicine

**Workflow:**
1. Manually score a PSG
2. Compare with automated results
3. Understand where you differ
4. Learn from discrepancies

---

## 🎨 Output Format

### Comparison Table
```
┌──────────────┬───────────┬────────┬────────────┬─────────┐
│ Metric       │ Automated │ Manual │ Difference │ % Diff  │
├──────────────┼───────────┼────────┼────────────┼─────────┤
│ AHI          │ 23.4      │ 24.1   │ -0.7       │ -2.9%   │
│ ODI          │ 18.2      │ 19.5   │ -1.3       │ -6.7%   │
│ Total Events │ 187       │ 193    │ -6         │ -3.1%   │
│ TST (hours)  │ 7.2       │ 7.3    │ -0.1       │ -1.4%   │
│ Baseline SpO₂│ 95.2      │ 95.5   │ -0.3       │ -0.3%   │
└──────────────┴───────────┴────────┴────────────┴─────────┘
```

### Agreement Metrics
```
Mean Absolute % Error: 2.9%
Overall Agreement: 🟢 Good
Metrics within 10%: 5/5
```

---

## 📥 Export Format (JSON)

```json
{
  "file": "patient_001.edf",
  "validation_date": "2026-02-09 14:30:00",
  "metrics_comparison": [
    {
      "Metric": "AHI",
      "Automated": "23.4",
      "Manual": "24.1",
      "Difference": "-0.7",
      "% Diff": "-2.9%"
    },
    ...
  ],
  "agreement_statistics": {
    "mape": 2.9,
    "agreement_level": "Good",
    "metrics_within_10pct": "5/5"
  }
}
```

---

## ⚠️ Common Discrepancy Causes

### 1. Different AASM Versions
- AASM 2007 vs 2012 vs 2017
- Different hypopnea criteria
- **Solution:** Document which version you're using

### 2. Event Detection Sensitivity
- Your app may detect more/fewer subtle events
- **Solution:** Adjust threshold parameters

### 3. Artifact Handling
- Different approaches to handling SpO₂ artifacts
- **Solution:** Match artifact filter settings

### 4. Baseline Calculation
- Different percentile methods (95th, 98th, etc.)
- **Solution:** Document your method clearly

### 5. Sleep Staging Differences
- YASA vs manual staging
- Affects TST calculation
- **Solution:** Validate staging separately first

### 6. Rounding Differences
- 23.4 vs 23 (clinical reports often round)
- **Solution:** Accept small (<1%) differences

---

## 🔍 Troubleshooting

### "Agreement shows Poor but metrics look close"
**Cause:** Small absolute differences on small values
**Example:** 2 events vs 3 events = 50% difference!
**Solution:** Focus on clinically important metrics (AHI, ODI)

### "AHI matches but events count differs"
**Cause:** Different TST calculations
**Solution:** Check TST comparison - likely the culprit

### "Everything differs significantly"
**Cause:** Possible channel mismatch or settings issue
**Solution:** Verify SpO₂ and flow channels are correct

---

## 📚 Interpretation Guide

### Good Agreement (MAPE < 10%)
**Interpretation:** Algorithm is performing well  
**Action:** Safe to use for research/clinical purposes  
**Confidence:** High

### Fair Agreement (MAPE 10-20%)
**Interpretation:** Reasonable performance, some expected variation  
**Action:** Acceptable for most uses, understand limitations  
**Confidence:** Moderate

### Poor Agreement (MAPE > 20%)
**Interpretation:** Significant discrepancies present  
**Action:** Investigate causes, may need manual review  
**Confidence:** Low - use with caution

---

## 🎯 Best Practices

### For Research Validation:
1. ✅ Use ≥10 files with diverse severity (mild, moderate, severe OSA)
2. ✅ Include healthy controls if possible
3. ✅ Calculate aggregate statistics (mean MAPE, correlation)
4. ✅ Report both central tendency and variability
5. ✅ Document all discrepancy causes

### For Clinical QA:
1. ✅ Spot-check 10-20% of automated analyses
2. ✅ Always review cases flagged as "Poor" agreement
3. ✅ Maintain log of validation results
4. ✅ Investigate systematic biases (always over/under)

### For Method Development:
1. ✅ Use fixed validation set across all method variations
2. ✅ Compare MAPE across methods
3. ✅ Document which method performs best
4. ✅ Consider clinical relevance, not just statistics

---

## 🚀 Future Enhancements (Coming Soon)

### Phase 2:
- [ ] CSV upload for epoch-by-epoch staging comparison
- [ ] Cohen's Kappa calculation for staging agreement
- [ ] Confusion matrix for stage classification
- [ ] Bland-Altman plots

### Phase 3:
- [ ] PDF report parsing (auto-extract metrics)
- [ ] Batch validation mode (multiple files at once)
- [ ] Event-by-event comparison (sensitivity/specificity)
- [ ] Population statistics across multiple validations

---

## 📖 References

**Statistical Methods:**
- Mean Absolute Percentage Error: [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
- Bland-Altman Analysis: Bland & Altman (1986) Lancet
- Cohen's Kappa: Cohen (1960) Educational and Psychological Measurement

**Sleep Scoring:**
- AASM Manual: [aasm.org](https://aasm.org/clinical-resources/scoring-manual/)
- Azarbarzin et al. (2019): Hypoxic Burden methodology

---

**Version:** 1.0  
**Last Updated:** February 9, 2026  
**Status:** Production Ready ✅
