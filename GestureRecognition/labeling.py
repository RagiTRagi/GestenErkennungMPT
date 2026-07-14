import os
import msvcrt
import numpy as np
import pickle
import shutil
import subprocess
from MPT_utils import augment_sequence


def data_labeling():
    """
     TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

     Ziel:
     -----
     Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
     Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess
       - Übergib einen Dateipfad für die Aufnahme
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    2. Interaktive Steuerung (optional)

        - Implementiere eine einfache Benutzerinteraktion:
        - Aufnahme speichern
        - Aufnahme verwerfen
        - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

          ESC → speichern
          andere Taste → verwerfen

    3. Daten sichten und bereinigen

        - Lade die aufgenommenen Daten
        - Überlege:
        - Welche Teile sind relevant?
        - Welche Frames sind leer oder unbrauchbar?
        - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
        - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    times : int
       Wie viele Aufnahmen gemacht werden sollen.
       Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).

    label : str
       Name der Geste / Klasse.
       Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).
    """
    # TO DO Input um label bei einem run wechseln zu können
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
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
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