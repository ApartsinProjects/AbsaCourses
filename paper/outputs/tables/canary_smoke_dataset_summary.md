| item | value | note |
| --- | --- | --- |
| Canary prefix | dataset_generation_canary_25_v5 | Accepted reference canary used for the pilot smoke test. |
| Batch ID | batch_69cc149df4c88190b7c0635bbfcb5d31 | Latest canary with accepted generation-quality gate. |
| Generated reviews | 10000 | Synthetic pilot dataset size used only for end-to-end pipeline validation. |
| Split sizes | 8000 / 1000 / 1000 | Train / validation / test split used by the benchmark harness. |
| Aspect inventory | 20 | Twenty-aspect protocol inventory enabled in the canary dataset contract. |
| Completed-rate gate | 1.00 | All 25 responses completed successfully. |
| Text-success gate | 1.00 | Every batch row returned parsable review text. |
| Duplicate-rate gate | 0.00 | No duplicate review texts were detected. |
| Length-match gate | 0.80 | Length-band adherence cleared the canary threshold. |
| Mean review length | 125.6 words | Observed average review length in the accepted canary output. |
| Acceptance verdict | True | This passing canary gated the fresh 10K submission. |
