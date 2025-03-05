from sklearn.datasets import fetch_california_housing

from evidently.future.datasets import Dataset
from evidently.future.report import Report
from evidently.future.metrics import *
from evidently.future.presets import DataDriftPreset

# Charger le dataset California Housing
data = fetch_california_housing(as_frame=True)
df = data['frame']

# Simuler des données de production en ajoutant un drift artificiel
train_data = df.sample(frac=0.7, random_state=42)  # 70% pour l'entraînement
prod_data = df.sample(frac=0.3, random_state=24)   # 30% pour la production

prod_data['MedInc'] *= 1.2
prod_data['AveOccup'] *= 1.7
ref = Dataset.from_pandas(train_data)
curent = Dataset.from_pandas(prod_data)
report = Report([DataDriftPreset()],include_tests=True)

my_eval = report.run(curent, ref)

my_eval.save_html("data/data_drift_report.html")