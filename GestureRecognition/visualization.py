import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def _split_sequences(dataset):
    """Zerlegt das flache ``X`` anhand von ``lengths`` in einzelne Trajektorien.

    Das Datensatzformat stammt aus :func:`labeling.dataset_building` und ist
    hmmlearn-kompatibel:

    - ``X``       : aneinandergehängte Punkte aller Sequenzen, shape ``(gesamt, 2)``
    - ``lengths`` : Anzahl Punkte pro Sequenz
    - ``y``       : ein Label pro Sequenz

    Returns
    -------
    list of (str, numpy.ndarray)
        Liste aus ``(label, trajektorie)``, wobei jede Trajektorie die
        shape ``(n_punkte, 2)`` hat.
    """
    X = np.asarray(dataset["X"], dtype=float)
    lengths = list(dataset["lengths"])
    labels = list(dataset["y"])

    sequences = []
    start = 0
    for length, label in zip(lengths, labels):
        sequences.append((label, X[start:start + length]))
        start += length
    return sequences


def visualize_dataset(dataset_path="dataset.pkl", max_per_class=5):
    """Visualisierung des eigenen Datensatzes (minimale erste Version).

    Lädt den von :func:`labeling.dataset_building` erzeugten Datensatz,
    zerlegt ihn wieder in einzelne Sequenzen und plottet pro Klasse mehrere
    Trajektorien übereinander. So lässt sich auf einen Blick prüfen, ob die
    Gesten klar unterscheidbar sind und ob es Ausreißer / leere Sequenzen gibt.

    Parameters
    ----------
    dataset_path : str, optional
        Pfad zum Datensatz-Pickle (Default ``"dataset.pkl"``).
    max_per_class : int, optional
        Wie viele Beispiel-Sequenzen je Klasse gezeichnet werden.

    Returns
    -------
    matplotlib.figure.Figure
        Die erzeugte Figur (praktisch für Tests / Weiterverarbeitung).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {dataset_path!r}. "
            "Erst mit labeling.dataset_building erzeugen."
        )

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Sequenzen nach Label gruppieren
    by_label = {}
    for label, seq in _split_sequences(dataset):
        by_label.setdefault(label, []).append(seq)

    labels = sorted(by_label)
    if not labels:
        raise ValueError("Datensatz enthält keine Sequenzen.")

    fig, axes = plt.subplots(
        1, len(labels), figsize=(4 * len(labels), 4), squeeze=False
    )
    for ax, label in zip(axes[0], labels):
        for seq in by_label[label][:max_per_class]:
            if len(seq) == 0:
                continue
            ax.plot(seq[:, 0], seq[:, 1], marker=".", markersize=2, alpha=0.7)
        ax.set_title(f"{label}  (n={len(by_label[label])})")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Bildkoordinaten: y zeigt nach unten

    fig.suptitle("Datensatz-Trajektorien pro Klasse")
    fig.tight_layout()
    plt.show()
    return fig


def evaluate_classifier(model_path="data/hmm.pkl", dataset_path="dataset.pkl"):
    """
    Evaluation des Klassifikators: Accuracy + Confusion Matrix.

    Laedt das trainierte Modell (dict: label -> hmm) und einen Testdatensatz
    ({X, y, lengths}), sagt fuer jede Sequenz eine Klasse voraus und vergleicht
    mit den echten Labels.
    """
    import pickle
    import numpy as np

    # Modell und Testdaten laden
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    X = np.array(dataset["X"])
    y = dataset["y"]
    lengths = dataset["lengths"]

    labels = list(model.keys())

    # Für jede Sequenz die Klasse mit dem besten Score vorhersagen
    y_true = []
    y_pred = []
    start = 0
    for i in range(len(lengths)):
        length = lengths[i]
        seq = X[start:start + length]
        start += length

        scores = {}
        for label in labels:
            scores[label] = model[label].score(seq)
        best_label = max(scores, key=scores.get)

        y_true.append(y[i])
        y_pred.append(best_label)

    # Accuracy = richtige / alle
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    accuracy = correct / len(y_true)
    print("Accuracy:", accuracy)

    # Confusion Matrix (Zeile = echtes Label, Spalte = vorhergesagtes Label)
    print("\nConfusion Matrix:")
    print("        " + "  ".join(labels))
    for t in labels:
        row = []
        for p in labels:
            count = 0
            for a, b in zip(y_true, y_pred):
                if a == t and b == p:
                    count += 1
            row.append(count)
        print(t, row)

    return accuracy


def replay_recordings():
    """
    TODO: Exploration und Replay der aufgenommenen Rohdaten

    Ziel:
    -----
    Ermögliche es, aufgenommene Sequenzen erneut abzuspielen
    und qualitativ zu überprüfen.

    Warum ist das wichtig?
    ----------------------
    - Du kannst überprüfen, ob deine Aufnahmen korrekt sind
    - Fehler in der Datenerfassung werden früh sichtbar
    - Du entwickelst ein besseres Verständnis für deine Daten

    Anforderungen / Ideen:
    ----------------------
    - Lade gespeicherte Aufnahmen
    - Spiele diese erneut ab (z. B. über SignalHub / Replay-Modus)
    - Iteriere über verschiedene Labels und Beispiele

    .. tip::
       Besonders hilfreich:
         - Vergleiche mehrere Beispiele derselben Klasse
         - Suche nach inkonsistenten Bewegungen

    .. warning::
       Schlechte oder inkonsistente Aufnahmen führen fast immer zu
       schlechten Modellen. Überprüfe deine Daten frühzeitig!

    Abgabe:
    -------
    - Du solltest zeigen können, wie deine Daten aussehen (Replay)
    - Du solltest erklären können:
        - Welche Beispiele gut sind
        - Welche problematisch sind

    Erweiterung (optional):
    -----------------------
    - Automatisches Filtern schlechter Sequenzen
    - Kombination mit Visualisierung
    """
    pass


if __name__ == "__main__":
    visualize_dataset()
