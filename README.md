# Dataset

- CSJ-APS (live recordings of academic presentation in public) and CSJ-SPS (live recordings of relaxed presentation)

- VCTK Corpus

- VIVOS

# Models

## RNN with CTC loss

```
python -m csp.train --config=ctc --dataset=aps
```

## seq2seq with attention mechanism

```
python -m csp.train --config=attention_char --dataset=aps
# or
python -m csp.train --config=attention_word --dataset=aps
```
