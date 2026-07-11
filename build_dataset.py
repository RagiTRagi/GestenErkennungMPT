# build_dataset.py - baut aus den recordings-Ordnern die Datensaetze
# Format: {"X": alle Punkte, "y": ein Label pro Sequenz, "lengths": Punkte pro Sequenz}
import os
import pickle
import numpy as np


def build(recordings_dir, output_path):
    """Liest einen recordings-Ordner ein und speichert einen Datensatz."""
    X = []
    y = []
    lengths = []

    for label in sorted(os.listdir(recordings_dir)):
        label_path = os.path.join(recordings_dir, label)
        if not os.path.isdir(label_path):
            continue

        for sample in os.listdir(label_path):
            with open(os.path.join(label_path, sample), "rb") as f:
                recording = pickle.load(f)

            # letzte gueltige Trajektorie aus der Aufnahme holen
            last_sequence = None
            for frame in recording["preprocessor"]:
                if frame is None or len(frame) == 0:
                    continue
                if frame["preprocessor"] is None:
                    continue
                last_sequence = frame["preprocessor"]

            # unbrauchbare (leere) Aufnahmen ueberspringen
            if last_sequence is None or len(last_sequence) == 0:
                print("uebersprungen (leer):", label, sample)
                continue

            last_sequence = np.asarray(last_sequence)
            y.append(label)
            lengths.append(len(last_sequence))
            X.extend(last_sequence)

    dataset = {"X": X, "y": y, "lengths": lengths}
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(output_path, "->", len(y), "Sequenzen,", len(set(y)), "Klassen")
    return dataset


# ein Datensatz pro Person + einer mit allen zusammen
ordner = {
    "recordings": "dataset.pkl",              # eigene Aufnahmen (Nam)
    "recordings_alex": "dataset_alex.pkl",
    "recordings_vincent": "dataset_vincent.pkl",
    "recordings_rafael": "dataset_rafael.pkl",
}

alle = {"X": [], "y": [], "lengths": []}
ohne_vincent = {"X": [], "y": [], "lengths": []}

for recordings_dir, output_path in ordner.items():
    if not os.path.isdir(recordings_dir):
        print("nicht gefunden, uebersprungen:", recordings_dir)
        continue
    dataset = build(recordings_dir, output_path)

    # alle zusammen
    alle["X"].extend(dataset["X"])
    alle["y"].extend(dataset["y"])
    alle["lengths"].extend(dataset["lengths"])

    # alle ausser Vincent
    if recordings_dir != "recordings_vincent":
        ohne_vincent["X"].extend(dataset["X"])
        ohne_vincent["y"].extend(dataset["y"])
        ohne_vincent["lengths"].extend(dataset["lengths"])

with open("dataset_all.pkl", "wb") as f:
    pickle.dump(alle, f)
print("dataset_all.pkl ->", len(alle["y"]), "Sequenzen (alle zusammen)")

with open("dataset_no_vincent.pkl", "wb") as f:
    pickle.dump(ohne_vincent, f)
print("dataset_no_vincent.pkl ->", len(ohne_vincent["y"]), "Sequenzen (ohne Vincent)")
