| item | value | note |
| --- | --- | --- |
| Production prefix | dataset_generation_10k_v2 | Current 10K generation package derived from the accepted v5 canary. |
| Batch ID | batch_69cc15c483488190941478aa4e3a976d | Completed production batch used for the benchmark in this revision. |
| Generated reviews | 10000 | Full synthetic corpus used in the present train-validation-test benchmark. |
| Split sizes | 8000 / 1000 / 1000 | Train / validation / test split used for all executed local baselines. |
| Aspect inventory | 20 | Twenty-aspect benchmark contract active in the production dataset. |
| Completed rate | 0.9159 | Some rows were truncated at max_output_tokens but still returned usable review text. |
| Text success rate | 1.0000 | Every production row returned parsable review text. |
| Duplicate rate | 0.0000 | No duplicate reviews were detected in the assembled dataset. |
| Length-match rate | 0.6819 | Length adherence at production scale was slightly weaker than the canary. |
| Mean review length | 117.2 words | Observed average review length in the assembled 10K corpus. |
| Usability verdict | usable with caveat | Benchmark proceeded because text coverage was complete despite incomplete status on some rows. |
