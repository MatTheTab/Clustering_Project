import pandas as pd
import yaml
import random
import sys
from umap import umap_ as UMAP

from pathlib import Path

params = yaml.safe_load(open('params.yaml'))['preprocess']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

Path('data/prepared2').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',')

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

umap_transformer = UMAP.UMAP(
    n_neighbors=params['umap']['n_neighbors'],
    n_components=params['umap']['n_components'],
    min_dist=params['umap']['min_dist']
)

df_umap = umap_transformer.fit_transform(df)

columns = [f"component_{x}" for x in range(1, params['umap']['n_components']+1)]

df_umap = pd.DataFrame(data = df_umap, columns = columns)

df_umap.to_csv(output_file, index=False)