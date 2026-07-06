from hmmlearn import hmm
import numpy as np
import pickle
import warnings

class HMMClassifier:
    """
    TODO: Implementiere einen HMM-basierten Klassifikator

    Ziel:
    -----
    Entwickle einen Klassifikator, der zeitliche Sequenzen mit Hilfe von
    Hidden-Markov-Modellen (HMMs) klassifiziert. Für HMMs können libraries wie
    :mod:`hmmlearn` benutzt werden

    Grundidee:
    ----------
    - Trainiere ein Modell pro Klasse
    - Bewerte neue Sequenzen anhand der Likelihood unter jedem Modell
    - Wähle die Klasse mit der höchsten Wahrscheinlichkeit

    .. note::
       Wie genau deine Modelle aussehen (z. B. Anzahl Zustände, Features,
       Initialisierung etc.) ist bewusst nicht vorgegeben.

    Wichtige Designentscheidungen:
    ------------------------------
    - Wie strukturierst du deine Trainingsdaten?
    - Wie repräsentierst du Sequenzen?
    - Wie verbindest du mehrere Sequenzen mit Labels?

    Speicherung:
    ------------
    Du solltest dir überlegen:
    - Wie speicherst du dein trainiertes Modell?
    - Wie lädst du es später wieder?
    - Welche Informationen müssen persistiert werden (z. B. Klassen, Modelle)?

    .. tip::
       ``pickle`` ist eine einfache Möglichkeit, Modelle zu speichern.
       Alternativ kannst du auch eigene Formate definieren.

    Evaluation:
    -----------
    Für sinnvolles Training solltest du unbedingt:
    - eine eigene ``train_test_split``-Logik implementieren
    - Trainings- und Testdaten sauber trennen

    .. warning::
       Wenn du Training und Test nicht trennst, sind deine Ergebnisse nicht aussagekräftig.

    Erweiterung (optional):
    -----------------------
    - Implementiere eine Grid Search für Hyperparameter
      (z. B. Anzahl Zustände, Modellstruktur)
    - Vergleiche verschiedene Modellkonfigurationen

    """
    def __init__(self, n_components=3, covariance_type="diag"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.models = {}  



    def fit(self, X, y, lengths):
        """
        TODO: Trainiere den Klassifikator

        Ziel:
        -----
        Trainiere ein separates HMM für jede Klasse basierend auf den
        gegebenen Sequenzen.


        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Daten so, dass du pro Klasse alle Sequenzen bekommst
        - Trainiere ein Modell pro Klasse
        - Speichere die trainierten Modelle intern

        .. tip::
           Überlege dir eine sinnvolle Datenstruktur wie:
           ``label -> (Daten, Sequenzlängen)``

        .. note::
           Die konkrete Umsetzung ist offen:
            - Wie genau du Daten aufteilst
            - Wie du dein Modell initialisierst
            - Welche Hyperparameter du verwendest

        .. warning::
           Achte darauf, dass:
            - ``lengths`` zu ``X`` passen
            - Labels korrekt zu Sequenzen zugeordnet sind

        Erweiterung:
        ------------
        - Experimentiere mit verschiedenen Modellgrößen
        - Nutze eine Grid Search zur Optimierung
        - Verwende ein separates Testset zur Evaluation

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lengths = np.asarray(lengths, dtype=int)

        # Basic input validation
        if lengths.sum() != X.shape[0]:
            raise ValueError("Sum of 'lengths' must equal number of rows in X")
        if len(y) != len(lengths):
            raise ValueError("Length of 'y' must equal length of 'lengths' (one label per sequence)")
        if np.any(lengths <= 0):
            raise ValueError("All sequence lengths must be positive integers")

        self.classes_ = np.unique(y)
        self.models = {}

        end_indices = np.cumsum(lengths)
        start_indices = np.insert(end_indices[:-1], 0, 0)

        sequences = [X[start:end] for start, end in zip(start_indices, end_indices)]

        for label in self.classes_:

            class_indices = np.where(y == label)[0]

            if len(class_indices) == 0:
                continue

            X_class_list = [sequences[i] for i in class_indices]
            lengths_label = [lengths[i] for i in class_indices]

            X_label = np.vstack(X_class_list)

            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=100,
                random_state=42
            )

            try:
                model.fit(X_label, lengths_label)
            except Exception as e:
                warnings.warn(f"HMM training failed for label {label}: {e}")
                # skip storing a model for this label
                continue

            self.models[label] = model
        
        return self

    def decision_function(self, X, lengths):
        """
        TODO: Berechne Scores für jede Klasse

        Ziel:
        -----
        Berechne für jede Eingabesequenz einen Score pro Klasse
        (z. B. Log-Likelihood unter jedem Modell).

        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Eingabe in einzelne Sequenzen
        - Berechne für jede Sequenz:
            Score unter jedem Klassenmodell
        - Gib eine Struktur zurück wie:
            ``(n_sequences, n_classes)``

        .. tip::
           Die meisten HMM-Implementierungen bieten eine
           ``score``-Funktion für Likelihoods.

        .. note::
           Du entscheidest selbst:
            - Welcher Score verwendet wird
            - Wie du mehrere Sequenzen behandelst

        .. warning::
           Stelle sicher, dass:
            - Die Reihenfolge der Klassen konsistent ist
            - Scores vergleichbar sind

        Returns
        -------
        scores : array-like
            Score pro Sequenz und Klasse
        """
        end_indices = np.cumsum(lengths)
        start_indices = np.insert(end_indices[:-1], 0, 0)

        sequences = [X[start:end] for start, end in zip(start_indices, end_indices)]
        
        n_sequences = len(sequences)
        n_classes = len(self.classes_)

        scores = np.zeros((n_sequences, n_classes))

        for i, seq in enumerate(sequences):
            for j, label in enumerate(self.classes_):
                model = self.models.get(label)
                if model is None:
                    seq_score = -np.inf
                else:
                    try:
                        seq_score = model.score(seq)
                    except Exception:
                        seq_score = -np.inf

                scores[i, j] = seq_score

        return scores

    def predict(self, X, lengths, return_scores=False):
        """
        TODO: Sage Klassenlabels voraus

        Ziel:
        -----
        Weise jeder Eingabesequenz ein Label zu.

        Anforderungen / Ideen:
        ----------------------
        - Nutze deine ``decision_function``
        - Wähle für jede Sequenz die Klasse mit bestem Score

        .. tip::
           Typischerweise:
           ``argmax über Klassen``

        .. note::
           Achte darauf, dass:
            - Klassenreihenfolge konsistent ist
            - Rückgabewerte klar interpretierbar sind

        Erweiterung:
        ------------
        - Gib zusätzlich Unsicherheiten oder Scores zurück
        - Implementiere Top-k Vorhersagen

        Returns
        -------
        labels : list
            Vorhergesagte Labels
        """
        scores = self.decision_function(X, lengths)
        best_indices = np.argmax(scores, axis=1)
        labels = [self.classes_[i] for i in best_indices]

        if return_scores:
            best_scores = np.max(scores, axis=1)
            return labels, best_scores
        else:
            return labels
        
    def save(self, filepath):
        """
        Speichere das trainierte Modell.
        Ziel:
        -----
        Speichere alle notwendigen Informationen, um das Modell später wiederherzustellen.
        """
        state = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "models": self.models,
            "classes_": self.classes_
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod 
    def load(cls, filepath):
        """
        Lade ein trainiertes Modell.
        Ziel:
        -----
        Rekonstruiere ein Modell aus einer gespeicherten Datei.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        instance = cls(
            n_components=state["n_components"],
            covariance_type=state["covariance_type"]
        )
        instance.models = state["models"]
        instance.classes_ = state["classes_"]
        return instance