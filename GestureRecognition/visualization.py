import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

from GestureRecognition.hmmclassifier import HMMClassifier


def visualize_dataset(dataset_path="dataset.pkl", max_per_class=5):
    """
    Zeigt die Trajektorien aus dem Datensatz an, ein Feld pro Buchstabe.

    So sieht man schnell, ob die Buchstaben unterscheidbar sind und
    ob es Ausreisser gibt.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    X = np.array(dataset["X"])
    y = dataset["y"]
    lengths = dataset["lengths"]

    # Datensatz wieder in einzelne Sequenzen zerschneiden und
    # nach Buchstabe gruppieren
    sequences = {}
    start = 0
    for i in range(len(y)):
        label = y[i]
        seq = X[start:start + lengths[i]]
        start = start + lengths[i]

        if label not in sequences:
            sequences[label] = []
        sequences[label].append(seq)

    labels = sorted(sequences.keys())

    # Raster ausrechnen: 6 Spalten, so viele Zeilen wie noetig
    ncols = 6
    nrows = len(labels) // ncols
    if len(labels) % ncols != 0:
        nrows = nrows + 1

    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for i in range(len(labels)):
        label = labels[i]
        ax = plt.subplot(nrows, ncols, i + 1)

        # mehrere Beispiele uebereinander zeichnen
        for seq in sequences[label][:max_per_class]:
            plt.plot(seq[:, 0], seq[:, 1], marker=".", markersize=2)

        plt.title(label + "  (n=" + str(len(sequences[label])) + ")")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Bildkoordinaten: y zeigt nach unten

    plt.suptitle("Datensatz-Trajektorien pro Klasse")
    plt.tight_layout()
    plt.show()


def evaluate_classifier(dataset_path="dataset.pkl", test_ratio=0.3, seed=42,
                        n_components=10):
    """
    Testet den Klassifikator: Accuracy + Confusion Matrix.

    Die Daten werden in Training und Test geteilt. Trainiert wird nur
    auf den Trainingsdaten, gemessen nur auf den Testdaten (die das
    Modell noch nie gesehen hat).
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    X = np.array(dataset["X"])
    y = dataset["y"]
    lengths = dataset["lengths"]

    # Datensatz in einzelne Sequenzen zerschneiden
    sequences = []
    start = 0
    for length in lengths:
        sequences.append(X[start:start + length])
        start = start + length

    # Train/Test-Split: von jedem Buchstaben kommt ein Teil in den Test
    random.seed(seed)
    train_idx = []
    test_idx = []
    for label in sorted(set(y)):
        # alle Indizes von diesem Buchstaben sammeln
        idx = []
        for i in range(len(y)):
            if y[i] == label:
                idx.append(i)

        random.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        if n_test == 0:
            n_test = 1

        test_idx = test_idx + idx[:n_test]
        train_idx = train_idx + idx[n_test:]

    # Trainingsdaten zusammenbauen
    X_train = []
    y_train = []
    len_train = []
    for i in train_idx:
        X_train.extend(sequences[i])
        y_train.append(y[i])
        len_train.append(len(sequences[i]))

    # Testdaten zusammenbauen
    X_test = []
    y_test = []
    len_test = []
    for i in test_idx:
        X_test.extend(sequences[i])
        y_test.append(y[i])
        len_test.append(len(sequences[i]))

    # Klassifikator NUR auf den Trainingsdaten trainieren
    clf = HMMClassifier(n_components=n_components)
    clf.fit(np.array(X_train), y_train, len_train)

    # auf den Testdaten vorhersagen
    y_pred = clf.predict(np.array(X_test), len_test)

    # Accuracy = richtige / alle
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct = correct + 1
    accuracy = correct / len(y_test)
    print("Test-Accuracy:", accuracy)

    # Confusion Matrix zaehlen (Zeile = echt, Spalte = vorhergesagt)
    labels = list(clf.classes_)
    cm = np.zeros((len(labels), len(labels)))
    for i in range(len(y_test)):
        row = labels.index(y_test[i])
        col = labels.index(y_pred[i])
        cm[row][col] = cm[row][col] + 1

    # als Heatmap anzeigen
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Echt")
    plt.title("Confusion Matrix (Test-Accuracy = " + str(round(accuracy, 2)) + ")")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return accuracy


def replay_recordings(recordings_dir="recordings", label=None, pause=0.03):
    """
    Spielt die aufgenommenen Buchstaben ab, Punkt fuer Punkt wie beim
    echten Schreiben. Mit label="A" wird nur ein Buchstabe abgespielt.
    """
    # alle Label-Ordner durchgehen (oder nur einen bestimmten)
    labels = sorted(os.listdir(recordings_dir))
    if label is not None:
        labels = [label]

    for lab in labels:
        label_path = os.path.join(recordings_dir, lab)
        files = os.listdir(label_path)

        # jede Aufnahme von diesem Buchstaben abspielen
        for file in files:
            filepath = os.path.join(label_path, file)
            with open(filepath, "rb") as f:
                recording = pickle.load(f)

            # letzte gueltige Trajektorie aus der Aufnahme holen
            traj = None
            for frame in recording["preprocessor"]:
                if frame is None or len(frame) == 0:
                    continue
                if frame["preprocessor"] is None:
                    continue
                traj = frame["preprocessor"]

            if traj is None or len(traj) == 0:
                print("uebersprungen (leer):", lab, "-", file)
                continue

            print("Replay:", lab, "-", file)

            xs = []
            ys = []
            for p in traj:
                xs.append(p[0])
                ys.append(p[1])

            # Punkt fuer Punkt nachzeichnen (immer 2 Punkte mehr)
            for i in range(2, len(xs) + 1, 2):
                plt.clf()
                plt.title(lab + " - " + file)
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
                plt.gca().invert_yaxis()  # Bildkoordinaten: y zeigt nach unten
                plt.plot(xs[:i], ys[:i], marker=".")
                plt.pause(pause)

            plt.pause(0.5)  # fertigen Buchstaben kurz stehen lassen

    plt.show()


if __name__ == "__main__":
    visualize_dataset()
