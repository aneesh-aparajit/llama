import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm
from datasets import load_dataset
import dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(name="corpus-generator")

dotenv.load_dotenv()


class CorpusGenerator:
    def __init__(self, dst_dir: str = "../../data/corpus/") -> None:
        self.dst_dir = dst_dir

    def run(
        self,
        dataset_path: str,
        hf_token: str,
        tgt_col: str,
        split: str = "train",
        out_file_name: str = "corpus.txt",
    ) -> None:
        try:
            logger.info("Starting downloading of data...")
            raw_dataset = load_dataset(dataset_path, "ml", token=hf_token)["train"]
            logger.info("Data downloaded...")
            with open(os.path.join(self.dst_dir, out_file_name), "w") as f:
                for _, row in tqdm(enumerate(raw_dataset)):
                    f.write(str(row[tgt_col]))
        except Exception as e:
            logger.error(e)
        return

    def __call__(self) -> None:
        parser = ArgumentParser(
            description="CLI for generating a corpus file from a  HuggingFace dataset repo"
        )
        parser.add_argument(
            "--path", type=str, required=True, help="Name of dataset you want to use."
        )
        parser.add_argument(
            "--column",
            type=str,
            required=True,
            help="Target column of the dataset you want to use for pretraining.",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="train",
            help="Dataset split (train/valid/test)",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            required=True,
            help="File name where you want to store the corpus file.",
        )
        args = parser.parse_args()

        self.run(
            dataset_path=args.path,
            hf_token=os.getenv("HF_TOKEN", None),
            tgt_col=args.column,
            split=args.split,
            out_file_name=args.output_file,
        )


if __name__ == "__main__":
    g = CorpusGenerator()
    g()
