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

# NLP1 Project 2: pretrained embeddings, embed-and-pool and LSTM on ISEAR

Embeddings: GloVe 6B (50/100/300d, from the [stanfordnlp/glove](https://huggingface.co/stanfordnlp/glove) mirror) and
[fastText wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html), tokenised with NLTK `word_tokenize`.
Encoders: mean/max pooling and (bi)LSTM, each followed by a one-hidden-layer softmax regressor.
Search: one-factor-at-a-time on a separate 5-fold split, then a second round combining the winners.
Evaluation: the same stratified 10-fold CV x 3 repeats as Project 1, Bayesian correlated t-test with a ROPE of 1 point,
and a Bayesian logistic model of per-text correctness for error analysis.

## Reproduce

```bash
mkdir -p embeddings
curl -L -o embeddings/glove.6B.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
curl -L -o embeddings/wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
pip install -r requirements.txt

python -m part2.explore
python -m part2.cv --folds 5 --runs 1 --seed 100 --out results/part2/search --configs pool pool-glove50 pool-glove100 pool-max pool-linear pool-nodropout pool-lr3e-3 pool-trainable pool-random pool-data25 pool-data50 lstm lstm-bi lstm-h64 lstm-h256 lstm-nodropout lstm-trainable lstm-random lstm-data25 lstm-data50
python -m part2.cv --folds 5 --runs 1 --seed 100 --out results/part2/search2 --configs pool-fasttext300 lstm-fasttext300 pool-trainable-nodropout pool-trainable-fasttext300 lstm-bi-h256 lstm-bi-fasttext300
python -m part2.cv --configs pool pool-trainable lstm lstm-bi-fasttext300 --out results/part2/final
python -m part2.analyze
python -m part2.errors
```

| file | |
|---|---|
| `part2/embeddings.py` | tokeniser, loading vectors, mapping texts to ids |
| `part2/explore.py` | coverage, neighbours, similarities, zero-shot check, analogies |
| `part2/models.py` | embed-and-pool / LSTM classifier and training with early stopping |
| `part2/cv.py` | all configurations and the cross-validation runner |
| `part2/analyze.py` | search and final comparisons, figures |
| `part2/errors.py` | features and Bayesian correctness model |
| `slides_part2.html` | backup slides (`slides_part2.pdf`) |
