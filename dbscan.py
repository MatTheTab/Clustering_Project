import pandas as pd
import yaml
import random
import sys
import pickle

from pathlib import Path

from sklearn.cluster import DBSCAN

params = yaml.safe_load(open('params.yaml'))['dbscan']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

model_file = Path('data') / 'models' / output_file

Path('data/models').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',', index_col=None)

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

model = DBSCAN(eps=params["epsilon"], min_samples=params["min_samples"], metric=params["metric"])
model.fit(df)

with open(model_file, 'wb') as f:
    pickle.dump(model, f)