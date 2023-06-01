import pandas as pd
import yaml
import random
import sys

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

params = yaml.safe_load(open('params.yaml'))['preprocess']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path('data') / 'prepared' / 'minmax_data.csv'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',')

if df.columns[0] == "id":
    df=df.drop("id",axis=1)
    
cols = df.columns

min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=[cols])

df_scaled.to_csv(output_file, index=False)