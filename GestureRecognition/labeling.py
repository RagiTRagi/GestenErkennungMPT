import os
import msvcrt
import numpy as np
import pickle
import shutil
import subprocess
from MPT_utils import augment_sequence


def data_labeling():
    """
    Nimmt Gestendaten über SignalHub auf und speichert ausgewählte Aufnahmen.

    Zu Beginn wird der Name der aufzunehmenden Geste über eine
    Benutzereingabe abgefragt. Dieser Name wird als Klassenlabel und als
    Name des zugehörigen Unterordners verwendet.

    Anschließend wartet die Funktion auf Tastatureingaben:

    - Mit der Leertaste wird eine neue Aufnahme gestartet.
    - Mit der Escape-Taste wird die Datenerfassung beendet.
    - Nach einer Aufnahme kann diese mit ``s`` gespeichert werden.
    - Mit einer anderen Taste, beispielsweise ``x``, wird die Aufnahme
      verworfen.

    Für jede Aufnahme wird SignalHub über einen Subprocess im
    Aufnahmemodus gestartet. SignalHub muss die erzeugten Daten unter
    ``record/test.pickle`` speichern.

    Beim Speichern wird die Aufnahme in einen label-spezifischen Unterordner
    innerhalb des Verzeichnisses ``recordings`` kopiert. Existiert der
    Unterordner noch nicht, wird er automatisch erstellt. Der Dateiname
    enthält das Klassenlabel sowie zwei zufällig erzeugte Zahlen, damit
    mehrere Aufnahmen derselben Geste nicht überschrieben werden.

    Returns
    -------
    None
        Die Funktion gibt keinen Wert zurück. Gespeicherte Aufnahmen werden
        als Pickle-Dateien im Aufnahmeverzeichnis abgelegt.

    Notes
    -----
    Die Funktion ist für Windows ausgelegt, da Tastatureingaben mit
    ``msvcrt.getch`` gelesen werden.

    SignalHub wird mit folgendem Befehl gestartet:

    ``uv run main.py --mode record``

    Die erwartete Ordnerstruktur nach dem Speichern ist beispielsweise:

    ``recordings/Herz/Herz_123456_7890.pickle``

    Die Funktion speichert die von SignalHub erzeugte Datei nicht direkt,
    sondern kopiert die temporäre Aufnahme aus ``record/test.pickle`` in
    das entsprechende Label-Verzeichnis.

    Raises
    ------
    FileNotFoundError
        Wenn ``uv``, ``main.py`` oder die temporäre Aufnahme
        ``record/test.pickle`` nicht gefunden wird.

    PermissionError
        Wenn das Aufnahmeverzeichnis nicht erstellt oder die Datei nicht
        dorthin kopiert werden darf.

    OSError
        Wenn beim Erstellen des Verzeichnisses, beim Starten des
        Subprocesses oder beim Kopieren der Aufnahme ein Betriebssystemfehler
        auftritt.
    """
    label = input("what label do you want to record?")
    while True:

        print("Click 'space' to record and 'esc' to end the program.")
        key = msvcrt.getch()
        if key == b"\x1b":  # esc für schließen
            break

        if key == b" ":
            print("Recording starts..")

            subprocess.run(["uv", "run", "main.py", "--mode", "record"])

            print("Save file press 's'\nDiscard file press 'x'")
            if msvcrt.getch() == b"s":
                cwd = os.getcwd()
                data_dir = os.path.dirname(os.path.join("..", cwd))
                folder = "recordings"
                data_path = os.path.join(data_dir, folder)

                random1 = np.random.randint(10000, 100000000)
                random2 = np.random.randint(1000, 1000000)
                filename = f"{label.title()}_{random1}_{random2}.pickle"

                try:
                    new_dir = os.path.join(data_path, label.title())
                    os.mkdir(new_dir)
                except FileExistsError:
                    print("Directory already exists.")

                filepath = os.path.join(new_dir, filename)
                source_path = "record/test.pickle"
                shutil.copy(source_path, filepath)
            else:
                continue

def dataset_building(output_path, recordings_dir="recordings", augment=0):
    """
    Erstellt einen HMM-Trainingsdatensatz aus gespeicherten Gestenaufnahmen.

    Die Funktion durchsucht den angegebenen Aufnahmeordner nach
    Unterordnern. Jeder Unterordner wird als eigenes Klassenlabel
    interpretiert. Die darin enthaltenen Pickle-Dateien werden geladen und
    nach gültigen Preprocessor-Sequenzen durchsucht.

    Aus jeder Aufnahme wird die letzte vorhandene und nicht leere
    Preprocessor-Sequenz übernommen. Alle Sequenzen werden anschließend
    hintereinander in einer gemeinsamen Feature-Liste gespeichert. Die
    ursprünglichen Sequenzlängen werden separat festgehalten, damit das
    Hidden-Markov-Modell die einzelnen Gesten trotz der zusammengeführten
    Daten voneinander unterscheiden kann.

    Optional können für jede gültige Originalsequenz zusätzliche,
    augmentierte Sequenzen erzeugt werden. Der fertige Datensatz wird als
    Pickle-Datei gespeichert.

    Parameters
    ----------
    output_path : str or os.PathLike
        Pfad, unter dem der erzeugte Datensatz als Pickle-Datei
        gespeichert wird.

    recordings_dir : str or os.PathLike, optional
        Verzeichnis mit den aufgenommenen Gestendaten. Jeder Unterordner
        entspricht einem Klassenlabel und enthält die zugehörigen
        Aufnahme-Dateien. Der Standardwert ist ``"recordings"``.

    augment : int, optional
        Anzahl der zusätzlich zu erzeugenden augmentierten Kopien pro
        gültiger Originalsequenz. Bei ``0`` wird keine Datenaugmentation
        durchgeführt. Der Standardwert ist ``0``.

    Returns
    -------
    dict
        Erzeugter Datensatz mit den folgenden Einträgen:

        ``"X"``
            Hintereinander gespeicherte Feature-Vektoren aller Sequenzen.

        ``"y"``
            Klassenlabel jeder Original- oder augmentierten Sequenz.

        ``"lengths"``
            Anzahl der Feature-Vektoren jeder einzelnen Sequenz. Die Summe
            der Werte entspricht der Gesamtzahl der Einträge in ``X``.

    Notes
    -----
    Die Funktion erwartet für jede Aufnahme ein mit ``pickle`` gespeichertes
    Dictionary mit dem Schlüssel ``"preprocessor"``. Dieser Schlüssel muss
    eine Folge von Frames enthalten. Jeder gültige Frame muss wiederum einen
    Eintrag namens ``"preprocessor"`` enthalten.

    Pro Aufnahme wird ausschließlich die letzte gültige und nicht leere
    Preprocessor-Sequenz verwendet. Vollständig leere oder ungültige
    Aufnahmen werden übersprungen.

    Die Datenaugmentation verwendet die Funktion ``augment_sequence``.
    Diese muss vor dem Aufruf dieser Funktion definiert oder importiert sein.

    Raises
    ------
    FileNotFoundError
        Wenn ``recordings_dir`` oder eine erwartete Aufnahme-Datei nicht
        existiert.

    KeyError
        Wenn eine geladene Aufnahme nicht die erwarteten
        ``"preprocessor"``-Einträge enthält.

    pickle.UnpicklingError
        Wenn eine Aufnahme-Datei keine gültige Pickle-Datei ist.
    """
    X = []
    y = []
    lengths = []

    for label in sorted(os.listdir(recordings_dir)):
        label_path = os.path.join(recordings_dir, label)
        if not os.path.isdir(label_path):
            continue

        for sample in os.listdir(label_path):
            filepath = os.path.join(label_path, sample)
            with open(filepath, "rb") as f:
                recording = pickle.load(f)

            
            last_sequence = None
            for frame in recording["preprocessor"]:
                if frame is None or len(frame) == 0:
                    continue
                if frame["preprocessor"] is None:
                    continue
                last_sequence = frame["preprocessor"]

            # leere Aufnahmen ueberspringen
            if last_sequence is None or len(last_sequence) == 0:
                print("uebersprungen (leer):", label, sample)
                continue

            last_sequence = np.asarray(last_sequence)
            y.append(label)
            lengths.append(len(last_sequence))
            X.extend(last_sequence)

            # Datenaugmentation: zusaetzliche leicht veraenderte Kopien
            for _ in range(augment):
                new_seq = augment_sequence(last_sequence)
                y.append(label)
                lengths.append(len(new_seq))
                X.extend(new_seq)

    dataset = {"X": X, "y": y, "lengths": lengths}
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(output_path, "->", len(y), "Sequenzen,", len(set(y)), "Klassen")
    return dataset


if __name__ == "__main__":
    # normaler Datensatz (fuer die Evaluation)
    dataset_building("dataset.pkl")
    # Datensatz mit Augmentation (fuers Training: pro Aufnahme 2 Kopien extra)
    dataset_building("dataset_augmented.pkl", augment=2)