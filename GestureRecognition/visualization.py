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


def evaluate_classifier():
    """
    TODO: Evaluation deines Klassifikators

    Ziel:
    -----
    Implementiere eine sinnvolle Auswertung deines Modells auf Testdaten.

    Warum ist das wichtig?
    ----------------------
    - Du brauchst objektive Metriken für die Qualität deines Modells
    - Training allein reicht nicht, entscheidend ist die Generalisierung

    Anforderungen / Ideen:
    ----------------------
    - Lade ein trainiertes Modell
    - Lade Testdaten (getrennt vom Training!)
    - Berechne Vorhersagen
    - Vergleiche Vorhersagen mit Ground Truth

    Metriken:
    ---------
    - Klassifikationsgenauigkeit (Accuracy)
    - Confusion Matrix

    .. tip::
       Eine Confusion Matrix zeigt dir:
         - Welche Klassen gut erkannt werden
         - Wo dein Modell Fehler macht

    .. warning::
       Testdaten dürfen **nicht** aus dem Training stammen!

    Interpretation:
    ---------------
    Du solltest erklären können:
    - Welche Klassen gut funktionieren
    - Welche Klassen verwechselt werden
    - Warum das passieren könnte

    .. note::
       Schlechte Performance liegt oft an:
         - schlechten Trainingsdaten
         - zu wenigen Beispielen
         - ungeeigneten Features

    Erweiterung (optional):
    -----------------------
    - Weitere Metriken (Precision, Recall, F1)
    - Vergleich verschiedener Modelle
    """
    pass


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
