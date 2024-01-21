import os
import sentencepiece as spm


class SentencePieceTrainer:
    def __init__(self) -> None:
        self.corpus_path = (
            "../../data/CulturaX/text/malayalam_pretraining_corpus_mini.txt"
        )
        self.output_dir = "../../checkpoints/malayalam-10k-mini/"
        self.model_prefix = "malayalam-10k"
        self.character_coverage = 1.0
        self.model_type = "bpe"
        self.vocab_size = 10000

    def train(self) -> str:
        output_path = os.path.join(self.output_dir, f"{self.model_prefix}.model")
        spm.SentencePieceTrainer.train(
            input=self.corpus_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            split_digits=True,
            allow_whitespace_only_pieces=True,
            byte_fallback=True,
            normalization_rule_name="identity",
            self_test_sample_size=0,
            input_format="text",
            unk_surface=r" \342\201\207 ",
            hard_vocab_limit=True,
            num_threads=os.cpu_count(),
        )

        os.rename(
            f"{self.model_prefix}.vocab",
            os.path.join(self.output_dir, f"{self.model_prefix}.vocab"),
        )
        os.rename(
            f"{self.model_prefix}.model",
            os.path.join(self.output_dir, f"{self.model_prefix}.model"),
        )

        return output_path

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.train()


if __name__ == "__main__":
    sp_trainer = SentencePieceTrainer()
    sp_trainer.run()
