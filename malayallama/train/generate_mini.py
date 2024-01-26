length = 18_000_000

from tqdm import tqdm

with open('../../data/corpus/culturaX_mini.txt', 'w') as f:
    for ix, line in tqdm(enumerate(open('../../data/corpus/culturaX.txt', 'r').readlines())):
        if ix == length:
            break
        else:
            f.write(line)
