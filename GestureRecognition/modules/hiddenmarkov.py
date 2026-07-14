import os
import pickle
from SignalHub import GALY, bgr, get_nested_key, Module
import numpy as np

from GestureRecognition.hmmclassifier import HMMClassifier


class HMMModule(Module):
    """Klassifiziert vorverarbeitete Gesten mit trainierten HMMs.

    Das Modul empfängt normalisierte Trajektorien über das Signal
    ``preprocessor``. Jede Trajektorie wird von den gespeicherten
    Klassenmodellen bewertet. Das Label mit der höchsten Log-Likelihood wird
    zusammen mit einer GALY-Visualisierung ausgegeben. Im Aufnahmemodus wird
    kein Modell geladen und keine Klassifikation durchgeführt.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """Registriert die Signale und speichert den Modellpfad.

        Parameters
        ----------
        outputSignal : str, optional
            Name des Signals, unter dem das erkannte Label ausgegeben wird.
        model_path : str, optional
            Pfad zu einem mit :meth:`HMMClassifier.save` gespeicherten Modell.
        **kwargs
            Zusätzliche Parameter für die Kompatibilität mit der
            Modulerzeugung. Sie werden derzeit nicht ausgewertet.
        """
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )

        self.outputSignal = outputSignal
        self.model_path = model_path

    def _draw_prediction_overlay(self, galy, best_label, best_probability):
        """Zeichnet das erkannte Label und dessen Konfidenz in ein Overlay.

        Parameters
        ----------
        galy : GALY
            Zeichenobjekt, dem die beiden Textzeilen hinzugefügt werden.
        best_label : object
            Label der Klasse mit dem höchsten Modellscore.
        best_probability : float
            Durch Softmax normalisierter Score des besten Labels im Bereich
            von 0 bis 1.

        Returns
        -------
        None
            Die Zeichenbefehle werden direkt in ``galy`` eingetragen.
        """
        x = 24
        y = 52

        title = str(best_label)
        subtitle = f"Confidence: {best_probability:.1%}"

        galy.putText(title, (x, y), fontScale=2.2, color=bgr("#fefcf7"), thickness=5)

        galy.putText(subtitle, (x, y + 42), fontScale=1.1, color=bgr("#ff500b"), thickness=3)

    def start(self, data):
        """Lädt die gespeicherten HMM-Klassenmodelle.

        Ist in der Konfiguration ``mode`` auf ``"record"`` gesetzt, wird das
        Laden übersprungen und ``self.model`` auf ``None`` gesetzt. In allen
        anderen Modi werden die Modelle und Klassenlabels des gespeicherten
        :class:`HMMClassifier` übernommen.

        Parameters
        ----------
        data : dict
            Framework-Daten mit dem optionalen Konfigurationswert ``mode``.

        Returns
        -------
        dict
            Leeres Dictionary, da während der Initialisierung kein Signal
            erzeugt wird.

        Raises
        ------
        FileNotFoundError
            Wenn außerhalb des Aufnahmemodus keine Modelldatei unter
            ``model_path`` vorhanden ist.
        """
        config = data.get("config", {})
        mode = get_nested_key("mode", config) if isinstance(config, dict) else None

        # letzte Prognose merken, damit sie dauerhaft im Bild steht
        self.last_text = None

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
        """Klassifiziert die aktuelle vorverarbeitete Trajektorie.

        Für jedes Klassenmodell wird die Log-Likelihood der Trajektorie
        berechnet. Die Scores werden mit einer Softmax-Transformation in
        relative Werte umgerechnet, die ausschließlich der Anzeige als
        Konfidenz dienen. Sie sind keine kalibrierten Wahrscheinlichkeiten.

        Parameters
        ----------
        data : dict
            Framework-Daten mit der normalisierten Trajektorie unter
            ``preprocessor``.

        Returns
        -------
        dict
            Bei einer gültigen Vorhersage das beste Label unter dem
            konfigurierten Ausgabesignal und das Text-Overlay unter ``galy``.
            Vor der ersten Vorhersage wird ein leeres Dictionary ausgegeben.
            In späteren Frames ohne neue Trajektorie enthält die Ausgabe das
            Label ``None`` und ein leeres GALY-Objekt.
        """

        trajectory = data["preprocessor"]

        best_label = None
        if trajectory is not None and self.model is not None:
            scores = {}
            for label, hmm in self.model.items():
                scores[label] = hmm.score(trajectory)

            best_label = max(scores, key=scores.get)

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
            self.last_text = f"{best_label}: {best_probability:.2%}"

        if self.last_text is None:
            return {}

        galy = GALY()
        if best_label is not None and best_probability is not None:
            self._draw_prediction_overlay(galy, best_label, best_probability)
        return {self.outputSignal: best_label, "galy": galy}

    def stop(self, data):
        """Beendet das Modul ohne zusätzliche Aufräumarbeiten.

        Parameters
        ----------
        data : dict
            Letzte Framework-Daten. Sie werden nicht verwendet.

        Returns
        -------
        None
        """
        pass
