# Detection Error By Gold Polarity

This report checks whether aspect detection depends on the gold sentiment polarity of the aspect mention. Detection recall here means: among gold aspect instances of a given polarity, how often was the aspect detected at all, regardless of whether the predicted sentiment was correct.

## albert-base-v2
- `negative`: gold 652, recall 1.0000, miss rate 0.0000, sentiment match given detected 0.0000
- `neutral`: gold 651, recall 1.0000, miss rate 0.0000, sentiment match given detected 1.0000
- `positive`: gold 710, recall 1.0000, miss rate 0.0000, sentiment match given detected 0.0000

## bert-base-uncased
- `negative`: gold 675, recall 0.4563, miss rate 0.5437, sentiment match given detected 0.0000
- `neutral`: gold 629, recall 0.2607, miss rate 0.7393, sentiment match given detected 1.0000
- `positive`: gold 680, recall 0.4985, miss rate 0.5015, sentiment match given detected 0.0000

## bert_joint
- `negative`: gold 652, recall 0.5813, miss rate 0.4187, sentiment match given detected 0.0000
- `neutral`: gold 651, recall 0.3641, miss rate 0.6359, sentiment match given detected 1.0000
- `positive`: gold 710, recall 0.5845, miss rate 0.4155, sentiment match given detected 0.0000

## distilbert-base-uncased
- `negative`: gold 675, recall 0.4400, miss rate 0.5600, sentiment match given detected 0.0000
- `neutral`: gold 629, recall 0.2798, miss rate 0.7202, sentiment match given detected 1.0000
- `positive`: gold 680, recall 0.4676, miss rate 0.5324, sentiment match given detected 0.0000

## distilbert_joint
- `negative`: gold 652, recall 0.5353, miss rate 0.4647, sentiment match given detected 0.0000
- `neutral`: gold 651, recall 0.3364, miss rate 0.6636, sentiment match given detected 1.0000
- `positive`: gold 710, recall 0.5380, miss rate 0.4620, sentiment match given detected 0.0000

## openai-gpt-5.4-zero-shot
- `negative`: gold 652, recall 0.3773, miss rate 0.6227, sentiment match given detected 0.8740
- `neutral`: gold 651, recall 0.1598, miss rate 0.8402, sentiment match given detected 0.0481
- `positive`: gold 710, recall 0.3592, miss rate 0.6408, sentiment match given detected 0.7490

## openai-gpt-5.4-zero-shot-glossary
- `negative`: gold 652, recall 0.3543, miss rate 0.6457, sentiment match given detected 0.9004
- `neutral`: gold 651, recall 0.1567, miss rate 0.8433, sentiment match given detected 0.0588
- `positive`: gold 710, recall 0.3577, miss rate 0.6423, sentiment match given detected 0.7283

## roberta-base
- `negative`: gold 652, recall 1.0000, miss rate 0.0000, sentiment match given detected 0.0000
- `neutral`: gold 651, recall 1.0000, miss rate 0.0000, sentiment match given detected 1.0000
- `positive`: gold 710, recall 1.0000, miss rate 0.0000, sentiment match given detected 0.0000

## tfidf_two_step
- `negative`: gold 675, recall 0.4489, miss rate 0.5511, sentiment match given detected 0.0000
- `neutral`: gold 629, recall 0.2798, miss rate 0.7202, sentiment match given detected 1.0000
- `positive`: gold 680, recall 0.4559, miss rate 0.5441, sentiment match given detected 0.0065

