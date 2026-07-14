from hmmlearn import hmm
import numpy as np
import pickle, os, sys, warnings, optuna
from sklearn.metrics import accuracy_score
from MPT_utils import augment_sequence, split_hmm_sequences_3way, plot_evaluation_results

class HMMClassifier:
    """
    Klassifikator für zeitliche Sequenzen auf Basis von Hidden-Markov-Modellen.

    Für jede vorhandene Klasse wird ein eigenes Gaussian Hidden Markov Model
    trainiert. Bei einer Vorhersage wird eine Eingabesequenz von allen
    Klassenmodellen bewertet. Als Vorhersage wird anschließend die Klasse
    gewählt, deren Modell die höchste Log-Likelihood für die Sequenz liefert.

    Die Modelle verwenden eine Left-to-Right-Initialisierung, auch
    Bakis-Topologie genannt. Dabei beginnt jede Sequenz im ersten versteckten
    Zustand. Ein Zustand kann zunächst entweder beibehalten oder zum direkt
    folgenden Zustand wechseln. Diese Struktur eignet sich insbesondere für
    zeitlich gerichtete Abläufe wie Gesten, Bewegungen oder Sprachsignale.

    Die Trainingsdaten werden in verketteter Form übergeben. ``X`` enthält
    alle Beobachtungen aller Sequenzen untereinander. Das Array ``lengths``
    beschreibt, wie viele aufeinanderfolgende Zeilen jeweils zu einer
    einzelnen Sequenz gehören. Das Array ``y`` enthält ein Klassenlabel pro
    Sequenz und nicht pro einzelner Beobachtung.

    Beispiel für drei Sequenzen::

        sequence_1 = np.array([...])  # Form: (20, n_features)
        sequence_2 = np.array([...])  # Form: (15, n_features)
        sequence_3 = np.array([...])  # Form: (24, n_features)

        X = np.vstack([sequence_1, sequence_2, sequence_3])
        lengths = [20, 15, 24]
        y = ["A", "B", "A"]

        classifier = HMMClassifier(n_components=5)
        classifier.fit(X, y, lengths)

    Parameters
    ----------
    n_components : int, default=10
        Anzahl der versteckten Zustände pro Klassenmodell. Eine größere Anzahl
        erlaubt komplexere zeitliche Abläufe, benötigt aber normalerweise auch
        mehr Trainingsdaten.

    covariance_type : {"spherical", "diag", "full", "tied"}, default="diag"
        Struktur der Kovarianzmatrizen der gaußschen Emissionsverteilungen.
        ``"diag"`` nimmt voneinander unabhängige Merkmale an und ist häufig
        stabiler, wenn nur begrenzte Trainingsdaten vorhanden sind.

    Attributes
    ----------
    models : dict
        Zuordnung von Klassenlabels zu den jeweils trainierten
        ``GaussianHMM``-Modellen.

    classes_ : numpy.ndarray
        Sortierte Klassenlabels, die während des Trainings in ``y`` gefunden
        wurden. Die Reihenfolge dieses Arrays bestimmt gleichzeitig die
        Spaltenreihenfolge der von :meth:`decision_function` zurückgegebenen
        Scores.

    Notes
    -----
    Die Klasse führt selbst keine Aufteilung in Trainings- und Testdaten
    durch. Die Sequenzen sollten deshalb vor dem Aufruf von :meth:`fit`
    beispielsweise klassenweise in Trainings- und Testsequenzen aufgeteilt
    werden. Einzelne Zeitpunkte derselben Sequenz dürfen dabei nicht auf
    Trainings- und Testdaten verteilt werden, da dies zu Data Leakage führen
    würde.

    """

    def __init__(self, n_components=10, covariance_type="diag"):

        """
        Initialisiere einen noch nicht trainierten HMM-Klassifikator.

        Beim Erstellen des Klassifikators werden noch keine HMMs erzeugt.
        Die Klassenmodelle werden erst in :meth:`fit` auf Grundlage der dort
        übergebenen Labels trainiert.

        Parameters
        ----------
        n_components : int, default=10
            Anzahl der versteckten Zustände jedes Klassenmodells.

        covariance_type : {"spherical", "diag", "full", "tied"}, default="diag"
            Art der Kovarianzmatrizendarstellung innerhalb der gaußschen
            Emissionsverteilungen.
        """

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.models = {}  

    def fit(self, X, y, lengths):

        """
        Trainiere für jede Klasse ein eigenes Gaussian-HMM.

        Die verketteten Beobachtungen aus ``X`` werden zunächst mithilfe von
        ``lengths`` wieder in einzelne Sequenzen zerlegt. Anschließend werden
        alle Sequenzen mit demselben Label zusammengefasst und zum Training
        eines gemeinsamen Klassenmodells verwendet.

        Jedes Klassenmodell wird mit einer Left-to-Right-Struktur
        initialisiert:

        - Eine Sequenz beginnt immer im ersten Zustand.
        - Jeder Zustand kann zunächst in sich selbst verbleiben.
        - Alternativ kann in den direkt folgenden Zustand gewechselt werden.
        - Der letzte Zustand besitzt zunächst nur einen Übergang zu sich
          selbst.

        Der Startzustand bleibt während des Trainings fest. Übergangsmatrix,
        Mittelwerte und Kovarianzen werden durch das HMM-Training optimiert.

        Parameters
        ----------
        X : array-like of shape (sum(lengths), n_features)
            Verkettete Beobachtungen aller Trainingssequenzen. Jede Zeile
            entspricht einem Zeitpunkt, jede Spalte einem Merkmal.

            Beispiele für Merkmale sind Gelenkkoordinaten, Winkel,
            Geschwindigkeiten oder aus Bildern extrahierte Embeddings.

        y : array-like of shape (n_sequences,)
            Klassenlabel für jede Sequenz. Die Labels können beispielsweise
            Zahlen oder Zeichenketten sein. Ein Eintrag in ``y`` gehört immer
            zu genau einem Eintrag in ``lengths``.

        lengths : array-like of shape (n_sequences,)
            Länge jeder Sequenz in Zeilen von ``X``. Die Summe aller Längen
            muss der Anzahl der Zeilen in ``X`` entsprechen.

            Beispiel::

                lengths = [20, 15, 24]

            bedeutet, dass die ersten 20 Zeilen von ``X`` zur ersten Sequenz,
            die nächsten 15 Zeilen zur zweiten Sequenz und die letzten
            24 Zeilen zur dritten Sequenz gehören.

        Returns
        -------
        self : HMMClassifier
            Der trainierte Klassifikator.

        Raises
        ------
        ValueError
            Wenn die Summe von ``lengths`` nicht der Anzahl der Beobachtungen
            in ``X`` entspricht.

        ValueError
            Wenn die Anzahl der Labels nicht der Anzahl der Sequenzen
            entspricht.

        ValueError
            Wenn mindestens eine Sequenzlänge kleiner oder gleich null ist.

        Warns
        -----
        UserWarning
            Wenn das Training eines einzelnen Klassenmodells fehlschlägt oder
            ungültige Wahrscheinlichkeiten erzeugt. Das betroffene Modell wird
            in diesem Fall nicht in ``models`` gespeichert. Andere Klassen
            werden dennoch weiter trainiert.

        Notes
        -----
        Die Qualität eines HMMs hängt stark von der Anzahl und Länge der
        verfügbaren Sequenzen ab. Insbesondere bei vielen Zuständen und nur
        wenigen Trainingssequenzen können Übergangs- oder
        Emissionswahrscheinlichkeiten instabil werden.

        Vor dem Training sollten die Merkmale gegebenenfalls normalisiert oder
        standardisiert werden. Eine dabei verwendete Transformation muss
        später für Test- und Live-Daten identisch angewendet werden.
        """

        X = np.asarray(X)
        y = np.asarray(y)
        lengths = np.asarray(lengths, dtype=int)

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
            class_indices = np.where(y == label)[0]
            if len(class_indices) == 0:
                continue

            X_class_list = [sequences[i] for i in class_indices]
            lengths_label = [lengths[i] for i in class_indices]
            X_label = np.vstack(X_class_list)

            startprob = np.zeros(self.n_components)
            startprob[0] = 1.0

            transmat = np.zeros((self.n_components, self.n_components))
            for i in range(self.n_components):
                if i == self.n_components - 1:
                    transmat[i, i] = 1.0  
                else:
                    transmat[i, i] = 0.5
                    transmat[i, i + 1] = 0.5

            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=100,
                random_state=42,
                init_params="mc", 
                params="tmc"  
            )
            
            model.startprob_ = startprob
            model.transmat_ = transmat

            try:
                model.fit(X_label, lengths_label)
                
                if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
                    raise ValueError("Das Modell hat NaNs generiert und ist kaputt.")
                    
            except Exception as e:
                warnings.warn(f"HMM training failed for label {label}: {e}")
                continue

            self.models[label] = model
        
        return self

    def decision_function(self, X, lengths):
        
        """
        Berechne die Klassen-Scores für eine oder mehrere Sequenzen.

        Jede Eingabesequenz wird von jedem trainierten Klassenmodell bewertet.
        Als Score wird die logarithmische Likelihood verwendet, welche durch
        :meth:`hmmlearn.hmm.GaussianHMM.score` berechnet wird.

        Ein hoher Wert bedeutet, dass die beobachtete Sequenz unter dem
        jeweiligen Klassenmodell vergleichsweise wahrscheinlich ist. Da es
        sich um Log-Likelihoods handelt, sind die Werte häufig negativ. Der
        numerisch größte Wert ist dennoch der beste Score.

        Parameters
        ----------
        X : array-like of shape (sum(lengths), n_features)
            Verkettete Beobachtungen der zu bewertenden Sequenzen. Die Anzahl
            und Reihenfolge der Merkmale muss mit den beim Training
            verwendeten Daten übereinstimmen.

        lengths : array-like of shape (n_sequences,)
            Anzahl der Beobachtungen pro Sequenz. Die Summe der Werte sollte
            der Anzahl der Zeilen in ``X`` entsprechen.

        Returns
        -------
        scores : numpy.ndarray of shape (n_sequences, n_classes)
            Log-Likelihood jeder Sequenz unter jedem Klassenmodell.

            Die Zeilen entsprechen den Eingabesequenzen. Die Spalten
            entsprechen den Klassen in der Reihenfolge von ``self.classes_``.
            Damit bezeichnet

            ``scores[i, j]``

            den Score der Sequenz ``i`` unter dem Modell der Klasse
            ``self.classes_[j]``.

        Notes
        -----
        Ist für eine Klasse kein erfolgreich trainiertes Modell vorhanden oder
        schlägt die Bewertung einer Sequenz fehl, wird für diese Kombination
        der Score ``-np.inf`` eingetragen.

        Die zurückgegebenen Werte sind Gesamt-Log-Likelihoods und hängen daher
        normalerweise von der Sequenzlänge ab. Der Vergleich verschiedener
        Klassen für dieselbe Sequenz ist sinnvoll. Scores unterschiedlich
        langer Sequenzen sind dagegen nicht ohne Weiteres direkt miteinander
        vergleichbar.
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
        Bestimme das wahrscheinlichste Klassenlabel jeder Sequenz.

        Zunächst werden mit :meth:`decision_function` die Log-Likelihoods
        aller Sequenzen unter allen Klassenmodellen berechnet. Für jede
        Sequenz wird anschließend die Klasse mit dem höchsten Score gewählt.

        Parameters
        ----------
        X : array-like of shape (sum(lengths), n_features)
            Verkettete Beobachtungen der vorherzusagenden Sequenzen.

        lengths : array-like of shape (n_sequences,)
            Länge jeder in ``X`` enthaltenen Sequenz.

        return_scores : bool, default=False
            Gibt an, ob zusätzlich zum vorhergesagten Label auch der höchste
            erreichte Klassen-Score zurückgegeben werden soll.

        Returns
        -------
        labels : list
            Vorhergesagtes Klassenlabel für jede Eingabesequenz.

        best_scores : numpy.ndarray of shape (n_sequences,), optional
            Höchste Log-Likelihood jeder Sequenz. Dieser Rückgabewert wird nur
            geliefert, wenn ``return_scores=True`` gesetzt wurde.

        Notes
        -----
        Die Scores stellen keine direkt normalisierten
        Klassenwahrscheinlichkeiten dar. Ein Score von beispielsweise ``-20``
        bedeutet daher nicht, dass die Klasse eine bestimmte prozentuale
        Wahrscheinlichkeit besitzt.

        Falls für eine Sequenz alle Klassen den Score ``-np.inf`` erhalten,
        liefert ``numpy.argmax`` technisch bedingt die erste Klasse aus
        ``self.classes_``. Dieser Sonderfall sollte bei einer produktiven
        Anwendung gegebenenfalls separat behandelt werden.
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
        Speichere den trainierten Klassifikator als Pickle-Datei.

        Persistiert werden sowohl die Konfiguration des Klassifikators als
        auch die trainierten Klassenmodelle und die Reihenfolge der
        Klassenlabels. Dadurch kann der Klassifikator später ohne erneutes
        Training mit :meth:`load` rekonstruiert werden.

        Parameters
        ----------
        filepath : str or path-like
            Zielpfad der zu erstellenden Datei, beispielsweise
            ``"models/gesture_hmm.pkl"``.

        Raises
        ------
        OSError
            Wenn die Datei am angegebenen Pfad nicht geschrieben werden kann.

        AttributeError
            Wenn der Klassifikator noch nicht trainiert wurde und deshalb das
            Attribut ``classes_`` nicht vorhanden ist.

        Notes
        -----
        Pickle-Dateien können von Python- und Bibliotheksversionen abhängig
        sein. Für eine spätere Reproduzierbarkeit sollten deshalb zusätzlich
        die verwendeten Versionen von Python, NumPy und hmmlearn dokumentiert
        werden.
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
        Lade einen zuvor gespeicherten HMM-Klassifikator.

        Die gespeicherten Einstellungen, Klassenlabels und HMM-Modelle werden
        aus einer Pickle-Datei gelesen. Anschließend wird eine neue
        ``HMMClassifier``-Instanz erzeugt und mit dem gespeicherten Zustand
        befüllt.

        Parameters
        ----------
        filepath : str or path-like
            Pfad zu einer Datei, die zuvor mit :meth:`save` erstellt wurde.

        Returns
        -------
        HMMClassifier
            Vollständig rekonstruierter und direkt für Vorhersagen
            verwendbarer Klassifikator.

        Raises
        ------
        OSError
            Wenn die angegebene Datei nicht gefunden oder nicht gelesen werden
            kann.

        KeyError
            Wenn die Datei nicht die erwarteten gespeicherten Bestandteile
            enthält.

        pickle.UnpicklingError
            Wenn die Datei keine gültige oder kompatible Pickle-Datei ist.

        Warning
        -------
        Pickle kann beim Laden beliebigen Python-Code ausführen. Es dürfen
        deshalb ausschließlich Dateien geladen werden, die aus einer
        vertrauenswürdigen Quelle stammen.
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


def main():

    """
    Trainiere oder evaluiere den HMM-basierten Sequenzklassifikator.

    Das Programm lädt einen serialisierten Datensatz, teilt die enthaltenen
    Sequenzen in Trainings-, Validierungs- und Testdaten auf und führt
    anschließend entweder eine reine Evaluation oder ein vollständiges
    Modelltraining durch.

    Der Datensatz muss als Pickle-Datei gespeichert sein und mindestens die
    folgenden Einträge enthalten:

    ``X``
        Verkettete Beobachtungen aller Sequenzen mit der Form
        ``(sum(lengths), n_features)``.

    ``y``
        Ein Klassenlabel pro Sequenz.

    ``lengths``
        Länge jeder einzelnen Sequenz. Die Summe aller Werte muss der Anzahl
        der Zeilen in ``X`` entsprechen.

    Standardmäßig wird die Datei ``dataset.pkl`` verwendet. Alternativ kann
    ein anderer Dateipfad als erstes Kommandozeilenargument angegeben werden::

        python hmm_classifier.py eigener_datensatz.pkl

    Mit dem Argument ``--eval`` wird kein neues Modell trainiert. Stattdessen
    wird das bereits gespeicherte Modell aus ``data/hmm.pkl`` geladen und auf
    dem Testdatensatz evaluiert::

        python hmm_classifier.py --eval

        python hmm_classifier.py eigener_datensatz.pkl --eval

    Beim vollständigen Training wird folgender Ablauf durchgeführt:

    1. Die Sequenzen werden im Verhältnis 70 Prozent Training,
       15 Prozent Validierung und 15 Prozent Test aufgeteilt.
    2. Ausschließlich die Trainingssequenzen werden augmentiert.
    3. Mit Optuna werden die Anzahl der versteckten Zustände und der
       Kovarianztyp optimiert.
    4. Die beste Konfiguration wird anhand der Validation Accuracy gewählt.
    5. Das finale Modell wird mit Trainings- und Validierungsdaten trainiert.
    6. Das Modell wird einmalig auf den unberührten Testdaten evaluiert.
    7. Das trainierte Modell wird unter ``data/hmm.pkl`` gespeichert.

    Die Testdaten werden weder zur Augmentation noch zur
    Hyperparameteroptimierung verwendet. Dadurch bleibt die abschließende
    Evaluation unabhängig von der Modellauswahl.
    """

    dataset_path = "dataset.pkl"
    eval_only = False

    if len(sys.argv) > 1:
        if "--eval" in sys.argv:
            eval_only = True
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

    print("Augmentiere Trainingsdaten (Faktor 2)...")
    
    end_indices = np.cumsum(lengths_train)
    start_indices = np.insert(end_indices[:-1], 0, 0)
    train_sequences = [X_train[start:end] for start, end in zip(start_indices, end_indices)]
    
    X_train_augmented = []
    y_train_augmented = []
    lengths_train_augmented = []

    for seq, label in zip(train_sequences, y_train):
        X_train_augmented.append(seq)
        y_train_augmented.append(label)
        lengths_train_augmented.append(len(seq))
        
        for _ in range(2):
            aug_seq = augment_sequence(seq)
            X_train_augmented.append(aug_seq)
            y_train_augmented.append(label)
            lengths_train_augmented.append(len(aug_seq))

    X_train = np.vstack(X_train_augmented)
    y_train = np.array(y_train_augmented)
    lengths_train = lengths_train_augmented

    print(f"Training nach Augmentation: {len(lengths_train)} Sequenzen")

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

    def objective(trial):

        """
        Bewerte eine von Optuna vorgeschlagene HMM-Konfiguration.

        Für jeden Trial werden die Anzahl der versteckten Zustände und der
        Kovarianztyp ausgewählt. Anschließend wird ein HMM-Klassifikator auf
        den augmentierten Trainingsdaten trainiert und auf den unabhängigen
        Validierungsdaten bewertet.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Aktueller Optuna-Trial, über den die zu testenden Hyperparameter
            vorgeschlagen werden.

        Returns
        -------
        float
            Accuracy des trainierten Klassifikators auf den
            Validierungssequenzen. Schlägt das Training oder die Vorhersage
            fehl, wird ``0.0`` zurückgegeben.
        """


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

    plot_evaluation_results(best_clf, X_test, y_test, lengths_test)

    os.makedirs("data", exist_ok=True)
    best_clf.save("data/hmm.pkl")
    print("Modell gespeichert unter data/hmm.pkl")

if __name__ == "__main__":
    main()