import pandas as pd
import yaml
import random
import sys

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

params = yaml.safe_load(open('params.yaml'))['preprocess']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

Path('data/prepared').mkdir(parents=True, exist_ok=True)
Path('data/prepared2').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',')

if df.columns[0] == "id":
    df=df.drop("id", axis=1)
if df.columns[0] == "Unnamed: 0":
    df=df.drop("Unnamed: 0", axis=1)

cols = df.columns

min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=[cols])

df_scaled.to_csv(output_file, index=False)