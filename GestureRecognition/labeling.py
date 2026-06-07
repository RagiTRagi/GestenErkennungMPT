import os
import msvcrt
from argparse import ArgumentParser
from SignalHub.configparser import ConfigParser
from SignalHub.engine import Engine, get_nested_key
from SignalHub.webcam import Webcam
from modules import *
from SignalHub.galyQT import qt_quit
import numpy as np#
import shutil

def data_labeling(times: int, label: str):
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
   
   if msvcrt.getch() == b' ':
      print("Recording starts..")

      modules = [
      #ConfigParser(parser),
      Webcam(),
      HandDetector(),
      TrailMarker(),
      Preprocessor(),
      HMMModule(),
      ]

      parser = ArgumentParser("GestureRecognition")
      parser.add_argument("--mode", action="store")
      parser.set_defaults(mode="record")
      config = ConfigParser(parser)
      data = config.start({"recorder": "replay"})
      engine = Engine(modules= modules, signals={})
      engine.run(data)

      print("Save file press 's'\nDelete file press 'x'")
      if msvcrt.getch() == b's':
         cwd = os.getcwd()
         data_dir = os.path.dirname(os.path.join("..", cwd))
         folder = "recordings"
         data_path = os.path.join(data_dir, folder)
         items = os.listdir(data_path)
         random1 = np.random.randint(10000, 100000000)
         random2 = np.random.randint(1000, 1000000)
         filename = f"{label.lower()}_{random1}_{random2}.pickle"
         print(filename)
         
         try:
            new_dir = os.path.join(data_path, label.title())
            os.mkdir(new_dir)
         except FileExistsError:
            print("Directory already exists.")
         filepath = os.path.join(new_dir,filename)
         print(filepath)
         source_path = get_nested_key("config.recorder.file", data)
         shutil.move(source_path, filepath)

         


      #if msvcrt.getch() == b"s":
      print(f"Saving file.. in {new_dir}")

      #print("Replay starts..")
      #parser2 = ArgumentParser("GestureRecognition")
      #parser2.set_defaults(mode="replay")
      #config2 = ConfigParser(parser2)
      #replay_data = config2.start({"recorder": "replay"})
      #print("Replay", replay_data)
      #engine = Engine(modules= modules, signals={})
      #engine.run(replay_data)
      #print("successful")
    
   
   return items

print(data_labeling(2, "herz"))


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
    pass