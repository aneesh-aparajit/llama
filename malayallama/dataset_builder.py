import os
import glob
import logging
import dask.dataframe as dd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self) -> None:
        self.src_dir = "../data/CulturaX/parquet/"
        self.dst_dir = "../data/CulturaX/text/"
    
    def dataframe(self, files: list[str]) -> dd.DataFrame:
        dfs = []
        for file in tqdm(files, total=len(files)):
            dfs.append(dd.read_parquet(path=file))
        return dd.concat(dfs)
    
    def run(self, output_file_name: str = "malayalam_pretraining_corpus.txt") -> None:
        try:
            os.makedirs(self.dst_dir, exist_ok=True)
            corpus_path = os.path.join(self.dst_dir, output_file_name)

            df = self.dataframe(glob.glob(os.path.join(self.src_dir, "*.parquet")))
            print(df.head())

            with open(corpus_path, "w") as f:
                for _, value in tqdm(df.iterrows(), total=len(df)):
                    f.write(str(value["text"]) + "\n")
        except Exception as e:
            logger.error(e)
        return corpus_path


if __name__ == '__main__':
    builder = DatasetBuilder()
    builder.run()
