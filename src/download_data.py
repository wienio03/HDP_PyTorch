from ucimlrepo import fetch_ucirepo
import pandas as pd

from logger import logger

# fetchind dataset from ucimlrepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# combining and saving the dataset
df = pd.concat([X, y], axis=1)
df.to_csv('data/raw/heart.csv', index=False)
logger.log(f'Downloaded dataset: {df.shape}')
logger.log(f'Columns: {list(df.columns)}')