"""
Schnelles, durchgehendes Aufnehmen von Buchstaben-Gesten.

Im Gegensatz zum alten Ablauf (pro Aufnahme ein neuer Prozess -> Webcam +
MediaPipe + Fenster jedes Mal neu laden) bleibt hier *eine* Session offen.
Webcam und Modell werden nur einmal geladen; danach koennen beliebig viele
Samples aufgenommen und der Buchstabe live gewechselt werden.

Aufruf:
    uv run GestureRecognition/record_letters.py
    uv run GestureRecognition/record_letters.py --label M   (Startbuchstabe)

Steuerung (das Kamera-Fenster muss fokussiert sein):
    S            aktuelle Trajektorie als Sample speichern (+ Puffer leeren)
    R            aktuelle Zeichnung verwerfen (gespeicherte Samples bleiben)
    Pfeil rechts naechster Buchstabe (A -> B -> ... -> Z -> A)
    Pfeil links  vorheriger Buchstabe
    Backspace    letztes gespeichertes Sample loeschen (undo)
    Esc          Fenster schliessen / beenden

Gespeichert wird nach:  <projekt>/recordings/<Buchstabe>/<Buchstabe>_..._....pickle
im gleichen Format, das dataset_building() in labeling.py erwartet.
"""

import argparse
import os
import sys

# Projekt-Wurzel auf den Importpfad legen, damit das Skript direkt lauffaehig ist
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SignalHub import ConfigParser, Engine, Webcam  # noqa: E402

from GestureRecognition.modules import HandDetector, Preprocessor, TrailMarker  # noqa: E402
from GestureRecognition.modules.labelrecorder import LabelRecorder  # noqa: E402

RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")


def main():
    parser = argparse.ArgumentParser("GestureRecognition-RecordLetters")
    # --mode muss existieren, damit die Engine kein None-Mode bekommt
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--label", action="store", default="A",
                        help="Buchstabe, mit dem gestartet wird (A-Z)")
    parser.add_argument("--webcam.width", required=False)

    # Modelldatei relativ zur Projekt-Wurzel finden, egal von wo gestartet wird
    os.chdir(PROJECT_ROOT)

    # Startlabel schon vor dem Engine-Parsing herausziehen
    known, _ = parser.parse_known_args()
    start_label = known.label

    preprocessor = Preprocessor()
    trailmarker = TrailMarker()

    modules = [
        ConfigParser(parser),
        Webcam(),
        HandDetector(),
        trailmarker,
        preprocessor,
        LabelRecorder(preprocessor, RECORDINGS_DIR,
                      start_label=start_label, trailmarker=trailmarker),
    ]

    engine = Engine(modules=modules, signals={})
    engine.run({})


if __name__ == "__main__":
    main()
