# NLP1 Project 1: emotion classification on ISEAR

7-way emotion classification (joy, fear, anger, sadness, disgust, shame, guilt) on
[ISEAR](https://huggingface.co/datasets/gsri-18/ISEAR-dataset-complete).

Features: token indicators, bag-of-words, TF-IDF over a BPE vocabulary trained on
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).
Models: log-linear, one-hidden-layer MLP, two-hidden-layer MLP.
Evaluation: stratified 10-fold CV x 3 repeats, Bayesian correlated t-test with a ROPE of 1 accuracy point.

## Reproduce

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python cv.py
for v in 1000 4000 16000 32000; do
  python cv.py --vocab $v --features tfidf --models loglin mlp --out results/vocab-$v
done
python analyze.py
python slides.py
```

| file | |
|---|---|
| `data.py` | loading and cleaning ISEAR |
| `tokenizer.py` | BPE training |
| `features.py` | indicator, BoW, TF-IDF |
| `models.py` | log-linear and MLPs |
| `cv.py` | cross-validation runs, writes `results/*/scores.csv` |
| `analyze.py` | Bayesian comparisons and figures |
| `slides.py` | builds `slides.pdf` |
