# train.py - trainiert die HMMs aus einem Datensatz und speichert das Modell
# fuer die Live-Demo (uv run main.py).
#
# Aufruf:
#   uv run train.py                        -> nutzt dataset.pkl (eigene Daten)
#   uv run train.py dataset_all.pkl        -> alle Personen zusammen
#   uv run train.py dataset_alex.pkl       -> nur Alex
import os
import pickle
import sys

from GestureRecognition.hmmclassifier import HMMClassifier

DATASET_PATH = "dataset.pkl"      # Standard: eigene Aufnahmen
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
MODEL_PATH = "data/hmm.pkl"       # genau der Pfad, den HMMModule live laedt

print("Trainiere mit:", DATASET_PATH)

# Datensatz laden
with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

X = dataset["X"]
y = dataset["y"]
lengths = dataset["lengths"]

# Klassifikator trainieren (ein HMM pro Buchstabe)
# 10 Zustaende: Buchstaben haben viele Bewegungsphasen, 3 war deutlich zu wenig
# (Test-Accuracy 0.38 -> 0.86)
clf = HMMClassifier(n_components=10)
clf.fit(X, y, lengths)
print("Trainierte Klassen:", clf.classes_)

# Modell speichern. hiddenmarkov.py laedt es ueber HMMClassifier.load(),
# deshalb mit clf.save() (volles Format: models + classes_ + Parameter).
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
clf.save(MODEL_PATH)

print("Modell gespeichert unter", MODEL_PATH)
