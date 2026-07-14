"""
Werkzeuge zur Sichtung des aufgenommenen Datensatzes.

Enthält Funktionen, um die Trajektorien des Datensatzes darzustellen,
die Güte des HMM-Klassifikators zu messen und einzelne Aufnahmen
abzuspielen.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from hmmclassifier import HMMClassifier
from MPT_utils import plot_evaluation_results, split_hmm_sequences_3way


def visualize_dataset(dataset_path="dataset.pkl", max_per_class=5):
    """
    Zeigt die Trajektorien des Datensatzes als Raster, ein Feld pro Buchstabe.

    Der Datensatz wird geladen, anhand von ``lengths`` wieder in einzelne
    Sequenzen zerschnitten und nach Buchstabe gruppiert. Pro Buchstabe
    werden bis zu ``max_per_class`` Beispiele übereinander gezeichnet.
    So sieht man auf einen Blick, ob die Buchstaben unterscheidbar sind
    und ob es Ausreißer gibt.

    Die y-Achse wird invertiert, da die Punkte in Bildkoordinaten
    vorliegen, y also nach unten zeigt.

    Parameters
    ----------
    dataset_path : str, optional
        Pfad zur Pickle-Datei mit den Schlüsseln ``X``, ``y`` und
        ``lengths``.
    max_per_class : int, optional
        Maximale Anzahl an Beispielen, die pro Buchstabe gezeichnet
        werden.
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

    # Raster ausrechnen: 6 Spalten, so viele Zeilen wie nötig
    ncols = 6
    nrows = len(labels) // ncols
    if len(labels) % ncols != 0:
        nrows = nrows + 1

    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for i in range(len(labels)):
        label = labels[i]
        ax = plt.subplot(nrows, ncols, i + 1)

        # mehrere Beispiele übereinander zeichnen
        for seq in sequences[label][:max_per_class]:
            plt.plot(seq[:, 0], seq[:, 1], marker=".", markersize=2)

        plt.title(label + "  (n=" + str(len(sequences[label])) + ")")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Bildkoordinaten: y zeigt nach unten

    plt.suptitle("Datensatz-Trajektorien pro Klasse")
    plt.tight_layout()
    plt.show()


def evaluate_classifier(dataset_path="dataset.pkl", model_path="data/hmm.pkl",
                        seed=179):
    """Wertet ein bereits trainiertes HMM auf dem Testdatensatz aus.

    Die Funktion stellt mit :func:`split_hmm_sequences_3way` denselben
    reproduzierbaren Testsplit wie das Trainingsskript her. Anschließend lädt
    sie das vorhandene Modell und übergibt ausschließlich die Testdaten an
    :func:`MPT_utils.plot_evaluation_results`. Es werden weder ein Modell
    trainiert noch Optuna-Trials ausgeführt.

    Parameters
    ----------
    dataset_path : str, optional
        Pickle-Datei mit den Schlüsseln ``X``, ``y`` und ``lengths``.
    model_path : str, optional
        Pickle-Datei des bereits trainierten :class:`HMMClassifier`.
    seed : int, optional
        Zufallswert des beim Training verwendeten Datensplits.

    Returns
    -------
    HMMClassifier
        Das geladene und ausgewertete Modell.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    (_, _, X_test, _, _, y_test,
     _, _, lengths_test) = split_hmm_sequences_3way(
        np.asarray(dataset["X"]), dataset["y"], dataset["lengths"],
        val_size=0.15, test_size=0.15, random_state=seed,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Kein trainiertes Modell unter '{model_path}' gefunden."
        )

    classifier = HMMClassifier.load(model_path)
    plot_evaluation_results(classifier, X_test, y_test, lengths_test)
    return classifier


def replay_recordings(recordings_dir="recordings", label=None, pause=0.03):
    """
    Spielt aufgenommene Buchstaben ab, Punkt für Punkt wie beim
    echten Schreiben.

    Durchläuft die Label-Ordner unter ``recordings_dir`` und lädt jede
    Aufnahme. Verwendet wird jeweils die letzte gültige Trajektorie einer
    Aufnahme, also der Stand am Ende des Schreibvorgangs; leere Aufnahmen
    werden übersprungen. Anschließend wird die Trajektorie in
    Zweierschritten nachgezeichnet.

    Parameters
    ----------
    recordings_dir : str, optional
        Ordner mit je einem Unterordner pro Buchstabe.
    label : str, optional
        Wird ein Buchstabe angegeben, z.B. ``"A"``, wird nur dieser
        abgespielt. Sonst werden alle abgespielt.
    pause : float, optional
        Wartezeit in Sekunden zwischen zwei Zeichenschritten.
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

            # letzte gültige Trajektorie aus der Aufnahme holen
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

            # Punkt für Punkt nachzeichnen (immer 2 Punkte mehr)
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
    evaluate_classifier("dataset.pkl", "data/hmm.pkl", seed=179)
