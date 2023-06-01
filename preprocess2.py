import pandas as pd
import yaml
import random
import sys
import umap.umap_ as umap

from pathlib import Path
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open('params.yaml'))['preprocess2']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
preprocessing_output = Path('data') / 'prepared' / 'preprocessed2_data.csv'

df = pd.read_csv(input_file, sep=',', index_col="id")
cols = df.columns

standard_scaler = StandardScaler()
df_scaled = standard_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=[cols])

umap_transformer = umap.UMAP(
    n_neighbors=params['umap_n_neighbors'],
    n_components=params['umap_n_components'],
    min_dist=params['umap_min_dist']
)

df_umap = umap_transformer.fit_transform(df_scaled)

columns = ["component " + str(x) for x in range(1, params["umap_n_components"]+1)]

df_umap = pd.DataFrame(data = df_umap, columns = columns)

df.to_csv(preprocessing_output)