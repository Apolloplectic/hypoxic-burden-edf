# Code Review: Hypoxic Burden Calculation Algorithm

## Executive Summary

This is a comprehensive review of a polysomnography (PSG) analysis application that calculates Hypoxic Burden metrics based on the Azarbarzin et al. (2019) methodology. The implementation shows a strong understanding of the methodology and correctly implements most aspects of the algorithm. However, there are several areas where improvements can be made to ensure full compliance with the published methodology and to enhance robustness.

## 1. CORRECTNESS ASSESSMENT

### 1.1 Azarbarzin 2019 Methodology Compliance

#### ✅ Correct Implementations:

1. **Reference Point**: The code correctly uses event START as the reference point for baseline calculation in `_calculate_stage_hb()` (line 737) and `calculate_bootstrap_ci()` (lines 1129, 1133-1135).

2. **Pre-event Baseline Window**: The default configuration correctly sets `pre_event_sec=100` (config.py line 15) and the implementation correctly extracts the 100s window before event start (analysis_engine.py lines 742-745).

3. **Desaturation Window**: The code correctly implements the desaturation window from event START to 90s AFTER event END (config.py line 17, analysis_engine.py lines 758-761, 1144-1147).

4. **Hypoxic Burden Formula**: The implementation correctly calculates HB = Σ(∫(baseline - SpO₂) dt) / TST where the integration is performed using trapezoidal rule (analysis_engine.py lines 772-773, 1158-1159).

#### ⚠️ Partial/Questionable Implementations:

1. **Baseline Method**: While the documentation claims to use MAXIMUM SpO₂ for baseline (analysis_engine.py line 750-753), there are inconsistencies:
   - In `_calculate_stage_hb()` and `calculate_bootstrap_ci()`, it correctly uses `baseline = base_df['spo2'].max()`
   - However, in `calculate_global_hypoxic_burden()` with 'stage_specific' and 'whole_night' methods, it uses `calculate_robust_baseline()` with 95th percentile (lines 902-904, 989-991), which contradicts the Azarbarzin method
   - The documentation in `_calculate_stage_hb()` (line 717) incorrectly states "98th percentile" instead of "maximum"

2. **Event Detection**: The event detection logic has some issues:
   - When using airflow, it validates events with SpO₂ desaturation using a 30s window AFTER event end (lines 358-361), which doesn't align with the Azarbarzin approach of analyzing from event start
   - When using SpO₂ only, it uses a rolling baseline with a 120s window (line 398) instead of the specified 100s pre-event window

### 1.2 Bugs and Logic Errors

1. **Baseline Calculation Inconsistency**: The global HB calculation uses percentile-based baselines instead of maximum, which doesn't align with the Azarbarzin method.

2. **Event Detection Misalignment**: The airflow-based event validation doesn't follow the Azarbarzin methodology for desaturation analysis.

3. **Documentation Error**: The comment in `_calculate_stage_hb()` (line 717) incorrectly states "98th percentile" instead of "maximum".

### 1.3 Deviations from Published Methodology

1. **Global HB Baseline Method**: The global HB calculation uses percentile-based baselines instead of the maximum method specified in Azarbarzin.

2. **Event Detection Approach**: The event detection validation doesn't follow the Azarbarzin methodology.

## 2. VALIDATION RECOMMENDATIONS

### 2.1 Verification Against Known Test Cases

To verify results against known test cases:

1. Create synthetic PSG data with known desaturation events
2. Manually calculate expected HB values using the Azarbarzin formula
3. Compare with algorithm output

### 2.2 Unit Tests

Key unit tests that would catch potential errors:

1. **Baseline Calculation Tests**: Verify that maximum SpO₂ is used for baseline in all methods
2. **Window Boundary Tests**: Ensure 100s pre-event and 90s post-event windows are correctly implemented
3. **Integration Tests**: Validate trapezoidal integration accuracy
4. **Edge Case Tests**: Handle missing data, boundary events, and artifacts
5. **Bootstrap CI Tests**: Verify stage-stratified sampling preserves event counts

### 2.3 Test Datasets

Suggestions for test datasets:

1. **Synthetic Data**: Generate artificial SpO₂ signals with controlled desaturations
2. **Public Datasets**: Use datasets from PhysioNet or other open repositories
3. **Edge Cases**: Include recordings with missing data, artifacts, and boundary events

## 3. OPTIMIZATION OPPORTUNITIES

### 3.1 Performance Improvements

1. **Vectorization**: The floating baseline calculation (lines 846-864) uses a loop that could be vectorized
2. **Memory Usage**: For long recordings, consider processing in chunks
3. **Bootstrap Optimization**: In `calculate_bootstrap_ci()`, avoid redundant calculations by precomputing baseline values

### 3.2 Code Clarity Improvements

1. **Consistent Naming**: Use consistent parameter names across methods
2. **Documentation**: Update comments to accurately reflect the implementation
3. **Modularization**: Consider separating different baseline calculation methods into distinct functions

### 3.3 Error Handling

1. **Division by Zero**: Add explicit checks for TST = 0
2. **Invalid SpO₂ Values**: Enhance validation for values outside physiological range
3. **Boundary Events**: Add specific handling for events at recording boundaries

## 4. CLINICAL RISK ASSESSMENT

### 4.1 Potential Misclassifications

1. **Baseline Miscalculation**: Using percentile instead of maximum baseline could lead to underestimation of HB
2. **Window Misalignment**: Incorrect time windows could miss significant desaturations
3. **Event Detection Issues**: Inaccurate event detection could affect overall HB calculation

### 4.2 Worst-Case Error Scenarios

1. **Severe Underestimation**: If baseline is calculated incorrectly, severe OSA could be reported as mild
2. **Missed Events**: Incorrect event detection could lead to false negatives
3. **Bootstrap CI Issues**: Invalid confidence intervals could mislead clinical interpretation

### 4.3 Safety Checks

1. **Baseline Validation**: Add sanity checks to ensure baseline values are within physiological range
2. **Area Validation**: Verify that calculated areas are non-negative
3. **Event Count Validation**: Ensure detected events are reasonable given recording duration

## Detailed Findings

### High Priority Issues

1. **Inconsistent Baseline Calculation** (analysis_engine.py):
   - The global HB calculation uses percentile-based baselines instead of maximum
   - This deviates from the Azarbarzin methodology and could significantly affect results
   - **Recommendation**: Modify `calculate_global_hypoxic_burden()` to use maximum baseline

2. **Documentation Error** (analysis_engine.py line 717):
   - Comment states "98th percentile" instead of "maximum"
   - **Recommendation**: Correct the comment to accurately reflect implementation

### Medium Priority Issues

1. **Event Detection Misalignment** (analysis_engine.py lines 358-361):
   - Airflow-based validation uses 30s window AFTER event end
   - This doesn't align with Azarbarzin's approach
   - **Recommendation**: Modify event detection to align with Azarbarzin methodology

2. **Performance Optimization** (analysis_engine.py lines 846-864):
   - Floating baseline calculation uses inefficient loop
   - **Recommendation**: Vectorize the calculation for better performance

### Low Priority Issues

1. **Parameter Validation**:
   - Add explicit checks for edge cases (division by zero, invalid values)
   - **Recommendation**: Enhance error handling throughout the codebase

## Recommendations

1. **Immediate Fixes**:
   - Correct the documentation error in `_calculate_stage_hb()`
   - Align global HB baseline calculation with Azarbarzin methodology
   - Fix event detection to align with Azarbarzin approach

2. **Short-term Improvements**:
   - Add comprehensive unit tests
   - Implement better error handling
   - Optimize performance-critical sections

3. **Long-term Enhancements**:
   - Develop a validation framework with synthetic test cases
   - Add more detailed logging for debugging
   - Consider implementing alternative methodologies for comparison

This implementation shows a solid foundation for hypoxic burden calculation but requires some adjustments to fully comply with the Azarbarzin 2019 methodology and ensure clinical accuracy.