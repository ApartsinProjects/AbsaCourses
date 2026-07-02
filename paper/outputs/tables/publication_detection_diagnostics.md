| family | approach | micro_f1 | macro_f1 | macro_balanced_accuracy | macro_specificity | macro_mcc | sentiment_mse_detected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local synthetic benchmark | bert-base-uncased | 0.2760 | 0.3364 | 0.6229 | 0.8050 | 0.2766 | 0.4959 |
| Local synthetic benchmark | distilbert-base-uncased | 0.2691 | 0.3376 | 0.6207 | 0.7863 | 0.2713 | 0.5044 |
| Local synthetic benchmark | distilbert_joint | 0.2524 | 0.3248 | 0.6131 | 0.7473 | 0.2609 | 0.5428 |
| Local synthetic benchmark | bert_joint | 0.2447 | 0.3208 | 0.6113 | 0.7020 | 0.2461 | 0.5288 |
| Local synthetic benchmark | tfidf_two_step | 0.2326 | 0.2867 | 0.5955 | 0.7225 | 0.1920 | 0.6830 |
| Local synthetic benchmark | albert-base-v2 | 0.1829 | 0.1828 | 0.5000 | 0.0000 | 0.0000 | 0.5773 |
| Local synthetic benchmark | roberta-base | 0.1829 | 0.1828 | 0.5000 | 0.0000 | 0.0000 | 0.6838 |
| GPT batch inference | openai-gpt-5.2-zero-shot | 0.2519 | 0.2417 | 0.5899 | 0.8686 | 0.1799 | 0.7179 |
| GPT batch inference | openai-gpt-5.2-retrieval-few-shot | 0.2501 | 0.2395 | 0.5883 | 0.8693 | 0.1823 | 0.7244 |
| GPT batch inference | openai-gpt-5.2-few-shot | 0.2450 | 0.2339 | 0.5848 | 0.8679 | 0.1798 | 0.7325 |
| GPT batch inference | openai-gpt-5.2-few-shot-diverse | 0.2374 | 0.2261 | 0.5800 | 0.8673 | 0.1653 | 0.7386 |
| Mapped real-data transfer | bert-base-uncased | 0.4593 | 0.3059 | 0.5925 | 0.8327 | 0.1874 | 0.3990 |
| Mapped real-data transfer | distilbert-base-uncased | 0.4156 | 0.3515 | 0.5778 | 0.7976 | 0.2182 | 0.3888 |
| Mapped real-data transfer | tfidf_two_step | 0.3740 | 0.2303 | 0.5403 | 0.7992 | 0.1162 | 0.7019 |
