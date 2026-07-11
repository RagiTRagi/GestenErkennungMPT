import os
import pickle
from SignalHub import GALY, bgr, get_nested_key, Module
import numpy as np

from GestureRecognition.hmmclassifier import HMMClassifier


class HMMModule(Module):
    """
    Modul zur Klassifikation von Gesten mittels Hidden Markov Models.

    Dieses Modul erhält eine vorverarbeitete Fingertrajektorie vom
    :class:`Preprocessor` Modul und verwendet ein trainiertes
    Hidden-Markov-Modell, um eine Geste zu klassifizieren.

    Ziel ist es, eine geladene Modellstruktur zu verwenden, um
    eine Entscheidung über die aktuell ausgeführte Bewegung zu treffen
    und das Ergebnis an das Framework zurückzugeben.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``preprocessor`` : normalisierte Trajektorien

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``markov``.

        Dieses Signal enthält Informationen über die erkannte Geste
        sowie deren Klassifikationsscore.

        Beispiel:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.

        model_path : str, optional
            Pfad zu einem gespeicherten HMM-Modell.

        **kwargs
            Weitere Parameter, die an :class:`Module` weitergegeben werden.
        """
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )

        self.outputSignal = outputSignal
        self.model_path = model_path

    def start(self, data):
        """
        Initialisierung des Moduls.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, ein zuvor trainiertes Hidden-Markov-Modell zu laden,
        das später zur Klassifikation verwendet wird.

        Hinweise
        --------
        - Das Modell kann aus einer Datei geladen werden.
        - Typischerweise wird dafür eine Klassenmethode verwendet,
          die ein gespeichertes Modell rekonstruiert.
        - Das geladene Modell sollte als Attribut des Moduls gespeichert
          werden, damit es in :meth:`step` verwendet werden kann.

        .. tip::
           Trenne klar zwischen:
            - Modell laden (``start``)
            - Modell anwenden (``step``)

        .. warning::
           Stelle sicher, dass:
            - der Pfad korrekt ist
            - das Modell zum erwarteten Datenformat passt

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        config = data.get("config", {})
        mode = get_nested_key("mode", config) if isinstance(config, dict) else None

        if mode == "record":
            self.model = None
            return {}

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Missing HMM model file: {self.model_path}. "
                "Train the model first or start the app in record mode."
            )

        loaded = HMMClassifier.load(self.model_path)
        self.model = loaded.models
        self.classes_ = loaded.classes_
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, eine vorverarbeitete Trajektorie zu klassifizieren
        und die wahrscheinlichste Geste zu bestimmen.

        Hinweise
        --------
        - Greife auf das ``preprocessor`` Signal zu.
        - Falls keine Trajektorie vorhanden ist, kann die Verarbeitung
          übersprungen werden.
        - Das geladene HMM-Modell kann anschließend verwendet werden,
          um eine Entscheidung für die aktuelle Bewegung zu berechnen.
        - Das Ergebnis enthält typischerweise Scores für mehrere Klassen.
        - Die Klasse mit dem höchsten Score kann als Ergebnis gewählt werden.

        Zusätzlich kann eine Visualisierung erzeugt werden:

        - Erzeuge ein :class:`GALY` Objekt.
        - Lege eine neue Zeichenebene an.
        - Verwende :meth:`putText`, um Score und Label darzustellen.
        - Für die Skalierung der Zeichenebene können Parameter aus der
          Konfiguration über :meth:`get_nested_key` gelesen werden.

        .. tip::
           Typischer Ablauf:
            1. Daten prüfen (existiert eine Sequenz?)
            2. Modell anwenden
            3. Scores interpretieren
            4. Ergebnis visualisieren

        .. note::
           Du entscheidest selbst:
            - wie du Scores darstellst
            - ob du nur das beste Label oder mehrere Kandidaten zeigst

        .. warning::
           Achte darauf, dass:
            - das Eingabeformat exakt zum Trainingsformat passt
            - keine leeren oder fehlerhaften Sequenzen verarbeitet werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``preprocessor`` : normalisierte Trajektorie
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Soll die erkannte Geste sowie optional Visualisierungsdaten
            enthalten.

            Beispiel:

            ``return {outputSignal: result, "galy": galy}``
        """

        trajectory = data["preprocessor"]

        if trajectory is None or self.model is None:
            return {}

        scores = {}
        scores = {}

        for label, hmm in self.model.items():
            scores[label] = hmm.score(trajectory)

        best_label = max(scores, key=scores.get)

        # Scores über Softmax in relative Wahrscheinlichkeiten umwandeln
        labels = list(scores.keys())
        score_values = np.array([scores[label] for label in labels])

        score_values = score_values - np.max(score_values)
        exp_scores = np.exp(score_values)
        probability_values = exp_scores / np.sum(exp_scores)

        probabilities = {
            label: float(probability)
            for label, probability in zip(labels, probability_values)
        }

        best_probability = probabilities[best_label]
        # for label, hmm in self.model.items():
        #    scores[label] = hmm.score(trajectory)
        # best_label = max(scores, key=scores.get)

        galy = GALY()
        # galy.putText(f"{best_label}: {scores[best_label]}", (20, 40))
        galy.putText(f"{best_label}: {best_probability:.2%}", (20, 40))
        return {self.outputSignal: best_label, "galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf interne Zustände zurückzusetzen
        oder Ressourcen freizugeben.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber relevant werden,
           wenn Modelle oder externe Ressourcen verwaltet werden.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass
