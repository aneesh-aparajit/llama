import dask.dataframe as dd
from collections import defaultdict
from googletrans import Translator
from tqdm import tqdm
import pandas as pd


translator = Translator(service_urls=['translate.googleapis.com'])

df = dd.read_parquet("../../data/airoboros/0000.parquet")

print(df.head())

translated = defaultdict(list)


for ix, row in tqdm(df.iterrows(), total=len(df)):
    if row['category'] != 'coding':
        translated['category'].append(row['category'])
        for data in row['conversations']:
            translated[data['from']].append(translator.translate(data['value'], src='en', dest='ml').text)

        pd.DataFrame(translated).to_csv("../../data/airoboros/malayalam.csv", index=False)
