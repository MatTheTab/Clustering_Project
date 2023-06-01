import pandas as pd
import yaml
import random
import sys

from pathlib import Path
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open('params.yaml'))['preprocess']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path('data') / 'prepared' / 'standardized_data.csv'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',', index_col="id")
cols = df.columns

standard_scaler = StandardScaler()
df_scaled = standard_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=[cols])

df_scaled.to_csv(output_file)