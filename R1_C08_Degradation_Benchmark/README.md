# Degradation benchmark

The common panel contains 127 filename-matched infrared–visible image pairs from DDL-12 (50), DeMMI-RF (20), EMS_lite (10), and the AWMM-100K snow subset (47). It contains 23 blur, 26 haze, 3 low-illumination, 16 noise, 4 overexposure, 8 rain, and 47 snow pairs.

As described in Section 5.6 and Table B1, BLIP-2 with OPT-2.7B generates descriptions from the degraded visible RGB image using four beams, up to 100 tokens, FP16, and decoding without sampling. The authors reviewed and corrected the descriptions before use. The captions are passed through each method's native text interface.

The final manifest and caption file use the same sample IDs as the metric records. Candidate IDs outside the final panel are listed in excluded_candidate_ids.csv. The source dataset totals are given in source_dataset_counts.csv. Table 8 reports condition-wise arithmetic means for SSIM, SCD, and SF. The per-image fusion file retains all eight primary metric values.
