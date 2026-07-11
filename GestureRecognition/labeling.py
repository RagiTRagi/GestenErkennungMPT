import os
import msvcrt
import numpy as np
import pickle
import shutil
import subprocess

# Projekt-Wurzel (eine Ebene ueber diesem Paket)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")


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


def dataset_building(output_path):
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
    y = []
    X = []
    lengths = []

    dataset = {"X": X, "y": y, "lengths": lengths}

    folder_path = RECORDINGS_DIR
    labels = os.listdir(folder_path)

    for label in labels:

        label_path = os.path.join(folder_path, label)
        samples = os.listdir(label_path)

        count = 0
        for sample in samples:
            count += 1
            trailmarker_sequence = []
            preprocessor_sequence = []
            y.append(label)

            filename = sample
            filepath = os.path.join(label_path, filename)

            with open(filepath, "rb") as f:
                loaded_pickle = pickle.load(f)

            preprocessor_data = []
            preprocessor = loaded_pickle["preprocessor"]

            for sequ in preprocessor:

                if sequ is None or len(sequ) == 0:
                    continue

                if sequ["preprocessor"] is None:
                    continue

                data = sequ["preprocessor"]
                preprocessor_data.append(data)
                last_sequence = preprocessor_data[-1]
            lengths.append(len(last_sequence))

            X.extend(last_sequence)

    print(dataset["lengths"])
    print(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    return None


if __name__ == "__main__":
    # Altes, interaktives Labeling (langsam: pro Aufnahme ein neuer Prozess).
    # Fuer schnelles Aufnehmen siehe GestureRecognition/record_letters.py
    dataset_path = os.path.join(PROJECT_ROOT, "dataset.pkl")
    print(dataset_building(dataset_path))
