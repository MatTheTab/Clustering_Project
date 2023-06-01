import pandas as pd
import yaml
import random
import sys

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

params = yaml.safe_load(open('params.yaml'))['davies_bouldin']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path('data') / 'prepared' / 'db_score.txt'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',', index_col=None)

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

cols = df.columns

k_range = range(params["range_x"], params["range_y"])

scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(df)
    score = davies_bouldin_score(df, labels)
    scores.append((k, score))   

best_score = min(scores, key=lambda x: x[1])[0]

with open(output_file, 'w') as of:
    of.write(str(best_score))