from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import yaml
import random
import sys
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))['preprocess']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
output_file = Path('data') / 'prepared' / 'pca_optimal_95.csv'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

data = pd.read_csv(input_file, sep=',')
data=data.drop(columns=["id"])

pca = PCA()
pca.fit(data)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

threshold = 0.95

optimal_n = np.argmax(cumulative_explained_variance_ratio >= threshold) + 1

pca = PCA(n_components=optimal_n)
transformed_data = pca.fit_transform(data)

columns = [f'PC{i+1}' for i in range(optimal_n)]
transformed_df = pd.DataFrame(transformed_data, columns=columns)

transformed_df.to_csv('transformed_data.csv', index=False)