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

model_file = 'data/models/dbscan.p'

Path('data/models').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',', index_col=None)

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

cols = df.columns

model = DBSCAN(eps=params["epsilon"], min_samples=params["min_samples"])
model.fit(df)

with open(model_file, 'wb') as f:
    pickle.dump(model, f)