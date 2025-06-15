from ucimlrepo import fetch_ucirepo
import pandas as pd

from src.logger import logger

# Fetching dataset from ucimlrepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Combining and saving the dataset
df = pd.concat([X, y], axis=1)
df.to_csv('data/raw/heart.csv', index=False)
logger.info(msg=f'Downloaded dataset: {df.shape}')
logger.info(msg=f'Columns: {list(df.columns)}')