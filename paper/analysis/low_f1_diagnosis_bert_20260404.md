# Low-F1 Diagnosis

## Dataset
- Rows: `10000`
- Mean words per review: `117.2`
- Mean aspects per review: `2.002`
- Duplicate texts: `0`

## Prediction Behavior
- Gold positive rate: `0.0992`
- Predicted positive rate: `0.1925`
- Macro average precision: `0.3459`
- Macro AUROC: `0.6756`
- Oracle-k micro-F1: `0.3347`

Most overpredicted aspects:
- `interest`: gold `0.101`, predicted `0.764`, delta `0.663`
- `overall_experience`: gold `0.1`, predicted `0.678`, delta `0.578`
- `support`: gold `0.096`, predicted `0.58`, delta `0.484`
- `accessibility`: gold `0.117`, predicted `0.342`, delta `0.225`
- `feedback_quality`: gold `0.09`, predicted `0.278`, delta `0.188`
- `prerequisite_fit`: gold `0.103`, predicted `0.203`, delta `0.1`
- `grading_transparency`: gold `0.105`, predicted `0.034`, delta `-0.071`
- `difficulty`: gold `0.104`, predicted `0.165`, delta `0.061`

Weakest aspects by ranking quality:
- `feedback_quality`: AP `0.114`
- `overall_experience`: AP `0.115`
- `support`: AP `0.1428`
- `clarity`: AP `0.185`
- `relevance`: AP `0.1908`
- `interest`: AP `0.198`
- `prerequisite_fit`: AP `0.2382`
- `difficulty`: AP `0.2675`

## Faithfulness
- Aspect support rate: `0.7705`
- Aspect sentiment-match rate: `0.4232`

Weakest support aspects:
- `relevance`: `0.25`
- `exam_fairness`: `0.6154`
- `practical_application`: `0.625`
- `accessibility`: `0.6333`
- `lecturer_quality`: `0.7273`
- `prerequisite_fit`: `0.7308`
- `grading_transparency`: `0.75`
- `materials`: `0.7857`

Weakest sentiment-match aspects:
- `relevance`: `0.2`
- `grading_transparency`: `0.25`
- `pacing`: `0.2667`
- `prerequisite_fit`: `0.3077`
- `tooling_usability`: `0.3667`
- `interest`: `0.4`
- `feedback_quality`: `0.4`
- `accessibility`: `0.4`
