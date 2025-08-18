# Validation Strategy Experiments

Documentation of 4 training experiments comparing different validation strategies for model selection.

**Date Range:** August 17, 2025  
**Configuration:** All experiments used identical architecture (256D hidden, 60D features, 24 batch size)  
**Goal:** Determine optimal validation strategy for structural understanding

---

## 🎯 **Experiment Summary**

| Strategy | Training Time | Line F1 | Chorus F1 | Boundary F1 | Complete Segments | V→C Accuracy |
|----------|---------------|---------|-----------|-------------|-------------------|--------------|
| **Boundary F1** | 37.2 min | 0.843 | 0.786 | **0.441** | 25.6% | 48.8% |
| **Composite** | 19.5 min | 0.806 | 0.731 | 0.438 | **28.9%** | **51.9%** |
| **WindowDiff** | 22.0 min | 0.787 | 0.693 | 0.374 | 19.5% | 38.0% |
| **Segment IoU** | 13.9 min | 0.773 | 0.687 | 0.377 | 26.2% | 48.8% |

---

## 📊 **Key Insights**

### 🥇 **Boundary F1 Strategy (WINNER)**
- **Best overall performance**: Highest boundary F1 (0.441) and line-level metrics
- **Longest training**: 37.2 minutes - model had time to optimize structural understanding
- **Balanced learning**: Strong both in line classification (84.3%) and boundary detection
- **Verdict**: Direct optimization of boundary detection produces best structural results

### 🥈 **Composite Strategy (RUNNER-UP)**  
- **Best transition detection**: Highest verse→chorus accuracy (51.9%) and complete segments (28.9%)
- **Efficient training**: Reasonable time (19.5 min) with good structural metrics
- **Balanced approach**: Combines multiple metrics effectively
- **Verdict**: Good balance between efficiency and structural understanding

### 🔶 **WindowDiff Strategy**
- **Conservative approach**: Focuses on forgiving boundary evaluation
- **Moderate results**: Middle-ground performance across all metrics
- **Early stopping**: Stopped earlier, missing potential improvements
- **Verdict**: Too forgiving - doesn't push model to achieve precise boundaries

### 🔶 **Segment IoU Strategy**
- **Fastest training**: Only 13.9 minutes - stopped too early
- **Segment focus**: Good complete segment detection (26.2%) despite short training
- **Underperformed**: Lowest line-level and boundary metrics
- **Verdict**: Early convergence prevented full potential

---

## 🎯 **Conclusions**

1. **Boundary F1 validation strategy is optimal** - directly optimizes the structural understanding we need
2. **Training time matters** - longer training (37+ min) allows architectural convergence  
3. **Composite strategy is efficient alternative** - best transition accuracy with reasonable training time
4. **Segment IoU stops too early** - needs patience parameter adjustment
5. **All strategies show similar boundary F1 range** (~0.37-0.44) - architectural limitations remain

---

## 🚀 **Recommendations**

**Primary:** Use `validation_strategy: "boundary_f1"` as default  
**Alternative:** Use `validation_strategy: "composite"` for faster iteration cycles  
**Next Steps:** Architectural improvements needed - all strategies plateau around 0.44 boundary F1

---

*These experiments validated Phase 5 completion and informed the prioritization of architectural enhancements (Phase 1-2) in the improvement plan.*
