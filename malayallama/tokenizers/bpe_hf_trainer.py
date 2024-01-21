from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

unk_token = "<UNK>"
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]


def prepare_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=spl_tokens)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


def train_tokenizer(files):
    tokenizer, trainer = prepare_tokenizer()
    tokenizer.train(files, trainer)
    tokenizer.save("./tokenizer-trained.json")
    tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
    return tokenizer


if __name__ == "__main__":
    files = glob("../../data/CulturaX/text/malayalam_pretraining_corpus.txt")
    tokenizer = train_tokenizer(files=files)
    print(tokenizer)
