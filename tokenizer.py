import string
from pathlib import Path

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

from data import load_tokenizer_corpus

UNK = "[UNK]"


def train_bpe(corpus, vocab_size=8000, min_frequency=5):
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = Metaspace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[UNK],
        initial_alphabet=list(string.ascii_lowercase + string.digits + string.punctuation),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def get_tokenizer(vocab_size=8000, cache="artifacts"):
    path = Path(cache) / f"bpe-{vocab_size}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = train_bpe(load_tokenizer_corpus(), vocab_size=vocab_size)
    tok.save(str(path))
    return tok
