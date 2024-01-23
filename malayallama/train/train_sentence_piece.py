import os
import sentencepiece as spm


class SentencePieceTrainer:
    def __init__(
        self,
        src_file: str,
        dst_dir: str,
        model_prefix: str = "malayalam-20k",
        vocab_size: int = 20_000,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
    ) -> None:
        self.src_file = src_file
        self.dst_dir = dst_dir
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type

    def train(self) -> None:
        output_path = os.path.join(self.src_dir, f"{self.model_prefix}.model")

        spm.SentencePieceTrainer.train(
            input=self.src_file,
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
            num_threads=2 * os.cpu_count(),
        )

        os.rename(
            f"{self.model_prefix}.vocab",
            os.path.join(output_path, f"{self.model_prefix}.vocab"),
        )
        os.rename(
            f"{self.model_prefix}.model",
            os.path.join(output_path, f"{self.model_prefix}.model"),
        )

        return output_path

    def __call__(self) -> None:
        pass
