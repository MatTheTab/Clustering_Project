import pandas as pd
import yaml
import random
import sys
import pickle

from pathlib import Path

from sklearn.cluster import KMeans

params = yaml.safe_load(open('params.yaml'))['kmeans']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

if params["silhouette_k"]:
    with open("data/scores/s_score.txt", 'r') as kf:
        k_clusters = int(kf.readline())
elif params["davies_bouldin_k"]:
    with open("data/scores/db_score.txt", 'r') as kf:
        k_clusters = int(kf.readline())
else:
    k_clusters = params["default_k"]

model_file = Path('data') / 'models' / output_file

Path('data/models').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',', index_col=None)

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

model = KMeans(n_clusters=k_clusters)
model.fit(df)

with open(model_file, 'wb') as f:
    pickle.dump(model, f)