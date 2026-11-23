# Benchmark settings and analysis records

The configuration files reproduce the evaluated settings and text interfaces specified in Tables 2, 3, and B1. Table 3 identifies the official reference revisions, and Table B1 lists the checkpoint filenames. The metric snapshot in code/metrics_v2.py documents the shared fusion metrics. Section 5.3 and Table 2 specify EvaNet's FMI_dct separately.

The [low-level results and rank and paired analysis scripts](../R1_C05_Low_Level_Results/README.md) and the [degradation evaluation records](../R1_C08_Degradation_Benchmark/README.md) also support this comment.

The text-control files provide raw CLIP ViT-B/32 and SigLIP Base/16 cosine scores for t1, t3, and t5 on the 20-pair LLVIP subset and per-image measurements for nine configurations in the 30-pair selected-caption versus native-null comparison.

The detection manifest identifies the RGBT-tiny training, validation, and test splits. The prediction files cover the 11 fused views of the 335-pair test set. Section 5.5 and Table 7 report the evaluation settings and comparisons with the VIS and IR controls. Section 5.7, Table 3, and Figure 10 specify the computational profiling protocol.
