# Low-level fusion results

The evaluation comprises 19 methods, using the same 30 TNO, 250 LLVIP, and 50 RoadScene image pairs. The eight primary metrics are EN, SD, SF, AG, MI, SSIM, CC, and SCD. The per-image values are on the original metric scales. The Table 4 file contains the arithmetic means at full precision and the values displayed in the manuscript. The Table 5 file gives differences relative to MetaFusion, 95% paired-bootstrap intervals, and Holm-adjusted results of two-sided paired permutation tests.

Figure 5 uses the normalized score s = 1 − (r − 1)/(M − 1), with M = 19 and average ranks for ties. Scores are averaged across the 24 dataset–metric combinations. The paired bootstrap resamples the matched image IDs within each dataset. Exceedance rates count strictly higher dataset-level means.

Run the scripts in the code folder with Python 3, NumPy, and pandas. Run analyze_rank_uncertainty.py for the normalized score analysis or stats_paired.py for the paired original-scale comparisons.
