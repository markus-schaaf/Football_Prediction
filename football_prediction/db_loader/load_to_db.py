import pandas as pd
from sqlalchemy import create_engine

# Parquet-Datei laden
input_file = 'football_prediction/data_lake/processed/matches_cleaned.parquet'
df = pd.read_parquet(input_file)

username = 'football_user'
password = 'simple123'
host = 'localhost'
port = '5432'
database = 'football_prediction'

# Verbindung aufbauen
connection_str = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
engine = create_engine(connection_str)

# Tabelle schreiben
df.to_sql('matches', engine, if_exists='replace', index=False)

