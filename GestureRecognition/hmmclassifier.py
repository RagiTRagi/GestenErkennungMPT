from hmmlearn import hmm
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
import optuna

from sklearn.metrics import accuracy_score

def augment_sequence(seq):
    """
    Macht aus einer Trajektorie eine leicht veraenderte Kopie:
    kleine Drehung + etwas Rauschen.
    """
    seq = np.asarray(seq)

    # kleine zufaellige Drehung (ca. -8 bis +8 Grad)
    angle = np.random.uniform(-0.15, 0.15)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = seq[:, 0] * cos_a - seq[:, 1] * sin_a
    y = seq[:, 0] * sin_a + seq[:, 1] * cos_a

    new_seq = np.column_stack([x, y])

    # etwas Rauschen auf jeden Punkt
    new_seq = new_seq + np.random.normal(0, 0.02, new_seq.shape)
    return new_seq

def split_hmm_sequences_3way(X, y, lengths, val_size=0.15, test_size=0.15, random_state=42):
    """
    Trennt Sequenzdaten in Train, Validation und Test Sets.
    """
    # 1. X anhand von 'lengths' in eine Liste von einzelnen Sequenzen aufteilen
    end_indices = np.cumsum(lengths)
    start_indices = np.insert(end_indices[:-1], 0, 0)
    sequences = [X[start:end] for start, end in zip(start_indices, end_indices)]
    
    # 2. Erster Split: Train+Val vs. Test
    # test_size ist der absolute Anteil für den Test (z.B. 15%)
    seq_temp, seq_test, y_temp, y_test = train_test_split(
        sequences, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Zweiter Split: Train vs. Validation
    # Wir müssen den val_size Anteil relativ zum verbleibenden temp-Datensatz berechnen
    relative_val_size = val_size / (1.0 - test_size)
    seq_train, seq_val, y_train, y_val = train_test_split(
        seq_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )
    
    # 4. Hilfsfunktion zum Zurückbauen in (X, lengths)
    def flatten(seqs):
        if not seqs:
            return np.array([]), []
        return np.vstack(seqs), [len(s) for s in seqs]

    X_train, len_train = flatten(seq_train)
    X_val, len_val = flatten(seq_val)
    X_test, len_test = flatten(seq_test)
    
    return (
        X_train, X_val, X_test, 
        np.array(y_train), np.array(y_val), np.array(y_test), 
        len_train, len_val, len_test
    )

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_evaluation_results(clf, X_test, y_test, lengths_test):
    """
    Visualisiert die Performance des optimierten Modells auf den Testdaten
    mithilfe einer Confusion Matrix.
    """
    # 1. Vorhersagen mit dem bereits trainierten Modell treffen
    y_pred = clf.predict(X_test, lengths_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Finale Evaluation ---")
    print(f"Test-Accuracy: {accuracy * 100:.2f}%")

    # 2. Confusion Matrix mit sklearn berechnen (viel robuster!)
    labels = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 3. Als Heatmap anzeigen
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    
    # Achsen beschriften
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    # Zahlen in die Kästchen schreiben (sehr praktisch!)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, int(cm[i, j]), 
                     ha="center", va="center", 
                     color="white" if cm[i, j] > (cm.max() / 2) else "black")

    plt.xlabel("Vorhergesagt")
    plt.ylabel("Echt")
    plt.title(f"Confusion Matrix (Test-Accuracy = {accuracy:.2f})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

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

    def __init__(self, n_components=10, covariance_type="diag"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.models = {}  
    def fit(self, X, y, lengths):
        X = np.asarray(X)
        y = np.asarray(y)
        lengths = np.asarray(lengths, dtype=int)

        # Basic input validation
        if lengths.sum() != X.shape[0]:
            raise ValueError("Sum of 'lengths' must equal number of rows in X")
        if len(y) != len(lengths):
            raise ValueError("Length of 'y' must equal length of 'lengths'")
        if np.any(lengths <= 0):
            raise ValueError("All sequence lengths must be positive integers")

        self.classes_ = np.unique(y)
        self.models = {}

        end_indices = np.cumsum(lengths)
        start_indices = np.insert(end_indices[:-1], 0, 0)
        sequences = [X[start:end] for start, end in zip(start_indices, end_indices)]

        for label in self.classes_:
            # ---------------------------------------------------------
            # 1. DATEN FÜR DIESE KLASSE EXTRAHIEREN (Hat vorher gefehlt!)
            # ---------------------------------------------------------
            class_indices = np.where(y == label)[0]
            if len(class_indices) == 0:
                continue

            X_class_list = [sequences[i] for i in class_indices]
            lengths_label = [lengths[i] for i in class_indices]
            X_label = np.vstack(X_class_list)

            # ---------------------------------------------------------
            # 2. Left-to-Right (Bakis) Initialisierung
            # ---------------------------------------------------------
            startprob = np.zeros(self.n_components)
            startprob[0] = 1.0

            transmat = np.zeros((self.n_components, self.n_components))
            for i in range(self.n_components):
                if i == self.n_components - 1:
                    transmat[i, i] = 1.0  # Letzter Zustand
                else:
                    transmat[i, i] = 0.5
                    transmat[i, i + 1] = 0.5

            # 3. Modell initialisieren
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=100,
                random_state=42,
                init_params="mc", 
                params="tmc"  # NEU: Das 's' fehlt! startprob_ wird nicht mehr geupdatet.
            )
            
            # Wir weisen unsere manuellen Matrizen zu
            model.startprob_ = startprob
            model.transmat_ = transmat

            try:
                model.fit(X_label, lengths_label)
                
                # NEU: Sicherheitscheck. Falls trotz "tmc" noch NaNs in der Übergangsmatrix 
                # entstehen (passiert bei sehr wenigen Daten), verwerfen wir das Modell.
                if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
                    raise ValueError("Das Modell hat NaNs generiert und ist kaputt.")
                    
            except Exception as e:
                warnings.warn(f"HMM training failed for label {label}: {e}")
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


if __name__ == "__main__":
    import os
    import sys
    import optuna
    from sklearn.metrics import accuracy_score
    import warnings

    dataset_path = "dataset.pkl"
    eval_only = False

    # Argumente parsen
    if len(sys.argv) > 1:
        # Wenn "--eval" übergeben wurde, springen wir direkt zur Visualisierung
        if "--eval" in sys.argv:
            eval_only = True
            # Falls zusätzlich ein Datensatz-Pfad übergeben wurde (der nicht "--eval" ist)
            remaining_args = [arg for arg in sys.argv[1:] if arg != "--eval"]
            if remaining_args:
                dataset_path = remaining_args[0]
        else:
            dataset_path = sys.argv[1]

    print("Lade Daten:", dataset_path)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    X_full = np.array(dataset["X"])
    y_full = dataset["y"]
    lengths_full = dataset["lengths"]

    print("Führe 3-Wege-Split durch (70% Train, 15% Val, 15% Test)...")
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, 
     lengths_train, lengths_val, lengths_test) = split_hmm_sequences_3way(
        X_full, y_full, lengths_full, val_size=0.15, test_size=0.15, random_state=42
    )

    # ---------------------------------------------------------
    # NEU: Nur die Trainingsdaten auf Sequence-Ebene augmentieren!
    # ---------------------------------------------------------
    print("Augmentiere Trainingsdaten (Faktor 2)...")
    
    # 1. X_train wieder in einzelne Sequenzen zerlegen
    end_indices = np.cumsum(lengths_train)
    start_indices = np.insert(end_indices[:-1], 0, 0)
    train_sequences = [X_train[start:end] for start, end in zip(start_indices, end_indices)]
    
    X_train_augmented = []
    y_train_augmented = []
    lengths_train_augmented = []

    # 2. Jede Trainingssequenz nehmen und Kopien hinzufügen
    for seq, label in zip(train_sequences, y_train):
        # Original behalten
        X_train_augmented.append(seq)
        y_train_augmented.append(label)
        lengths_train_augmented.append(len(seq))
        
        # 2 augmentierte Kopien erstellen
        for _ in range(2):
            aug_seq = augment_sequence(seq) # Nutze deine Funktion aus dem anderen File
            X_train_augmented.append(aug_seq)
            y_train_augmented.append(label)
            lengths_train_augmented.append(len(aug_seq))

    # 3. Wieder für das HMM-Training flachklopfen
    X_train = np.vstack(X_train_augmented)
    y_train = np.array(y_train_augmented)
    lengths_train = lengths_train_augmented

    print(f"Training nach Augmentation: {len(lengths_train)} Sequenzen")

    # ---------------------------------------------------------
    # MODUS 1: Nur Evaluation (Schnelltest)
    # ---------------------------------------------------------
    if eval_only:
        model_path = "data/hmm.pkl"
        if not os.path.exists(model_path):
            print(f"Fehler: Kein trainiertes Modell unter '{model_path}' gefunden! Bitte erst einmal normal trainieren.")
            sys.exit(1)
            
        print(f"Lade existierendes Modell aus {model_path} für Schnelltest...")
        loaded_clf = HMMClassifier.load(model_path)
        
        print("Generiere Confusion Matrix...")
        plot_evaluation_results(loaded_clf, X_test, y_test, lengths_test)
        print("Schnelltest beendet.")
        sys.exit(0)

    # ---------------------------------------------------------
    # MODUS 2: Volles Training + Optuna (Dein bisheriger Code)
    # ---------------------------------------------------------
    def objective(trial):
        n_components = trial.suggest_int("n_components", 3, 12)
        covariance_type = trial.suggest_categorical("covariance_type", ["spherical", "diag", "full", "tied"]) 
        
        clf = HMMClassifier(n_components=n_components, covariance_type=covariance_type)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_train, y_train, lengths_train)
            except Exception:
                return 0.0

        try:
            y_pred = clf.predict(X_val, lengths_val)
            return accuracy_score(y_val, y_pred)
        except Exception:
            return 0.0

    print("Starte Optuna Hyperparameter-Suche...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=True)

    print("\n--- Optuna Ergebnisse ---")
    print("Beste Parameter:", study.best_params)
    print(f"Beste Validation Accuracy: {study.best_value * 100:.2f}%")

    print("\nTrainiere finales Modell mit besten Parametern...")
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    lengths_train_val = lengths_train + lengths_val

    best_clf = HMMClassifier(
        n_components=study.best_params["n_components"],
        covariance_type=study.best_params["covariance_type"]
    )
    best_clf.fit(X_train_val, y_train_val, lengths_train_val)

    # Plot und Speichern
    plot_evaluation_results(best_clf, X_test, y_test, lengths_test)

    os.makedirs("data", exist_ok=True)
    best_clf.save("data/hmm.pkl")
    print("Modell gespeichert unter data/hmm.pkl")