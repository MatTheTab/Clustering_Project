import pandas as pd
import yaml
import random
import sys

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

params = yaml.safe_load(open('params.yaml'))['preprocess1']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
preprocessing_output = Path('data') / 'prepared' / 'preprocessed1_data.csv'

df = pd.read_csv(input_file, sep=',', index_col="id")
cols = df.columns

min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=[cols])

pca = PCA(n_components=params["pca"])
df_pca = pca.fit_transform(df_scaled)

columns = ["PC" + str(x) for x in range(1, params["pca"]+1)]

df_pca = pd.DataFrame(data = df_pca, columns = columns)

df.to_csv(preprocessing_output)