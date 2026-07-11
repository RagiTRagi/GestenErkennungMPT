"""
Schnelles, durchgehendes Aufnehmen von Buchstaben-Gesten.

Im Gegensatz zum alten Ablauf (pro Aufnahme ein neuer Prozess -> Webcam +
MediaPipe + Fenster jedes Mal neu laden) bleibt hier *eine* Session offen.
Webcam und Modell werden nur einmal geladen; danach koennen beliebig viele
Samples aufgenommen und der Buchstabe live gewechselt werden.

Aufruf:
    uv run GestureRecognition/record_letters.py
    uv run GestureRecognition/record_letters.py --label M   (Startbuchstabe)
    uv run GestureRecognition/record_letters.py --review    (Aufnahmen durchschauen/loeschen)
    uv run GestureRecognition/record_letters.py --review --label M

Steuerung Aufnahme (das Kamera-Fenster muss fokussiert sein):
    S            aktuelle Trajektorie als Sample speichern (+ Puffer leeren)
    R            aktuelle Zeichnung verwerfen (gespeicherte Samples bleiben)
    Pfeil rechts naechster Buchstabe (A -> B -> ... -> Z -> A)
    Pfeil links  vorheriger Buchstabe
    Backspace    letztes gespeichertes Sample loeschen (undo)
    Esc          Fenster schliessen / beenden

Steuerung Review (--review, Plot-Fenster fokussiert):
    Pfeil rechts/links   naechstes / vorheriges Sample
    Pfeil hoch/runter    naechster / vorheriger Buchstabe
    D oder Entf          angezeigtes Sample LOESCHEN (danach neu aufnehmen)
    Q oder Esc           Review beenden

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

ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def review_recordings(start_label="A", recordings_dir=RECORDINGS_DIR):
    """
    Vorhandene Aufnahmen durchschauen und schlechte direkt loeschen.

    Zeigt pro Aufnahme die Trajektorie (gruener Punkt = Start, rotes
    Quadrat = Ende). Geloeschte Samples koennen danach im normalen
    Aufnahme-Modus einfach neu gemacht werden.
    """
    import pickle

    import numpy as np
    import matplotlib.pyplot as plt

    state = {
        "letter": ALPHABET.index(start_label) if start_label in ALPHABET else 0,
        "i": 0,
        "info": "",
    }

    def files_for(letter):
        d = os.path.join(recordings_dir, letter)
        if not os.path.isdir(d):
            return []
        return sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pickle")
        )

    def load_traj(path):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        traj = None
        for frame in rec["preprocessor"]:
            if frame is None or len(frame) == 0:
                continue
            if frame["preprocessor"] is None:
                continue
            traj = frame["preprocessor"]
        return None if traj is None else np.asarray(traj)

    fig, ax = plt.subplots(figsize=(6, 6.5))

    def draw():
        ax.clear()
        letter = ALPHABET[state["letter"]]
        files = files_for(letter)

        if not files:
            ax.set_title(f"{letter}: keine Aufnahmen vorhanden")
        else:
            state["i"] %= len(files)
            path = files[state["i"]]
            traj = load_traj(path)
            if traj is not None and len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], marker=".", markersize=3)
                ax.plot(traj[0, 0], traj[0, 1], "go", markersize=10)   # Start
                ax.plot(traj[-1, 0], traj[-1, 1], "rs", markersize=8)  # Ende
            ax.set_title(
                f"{letter}   Sample {state['i'] + 1}/{len(files)}\n"
                f"{os.path.basename(path)}"
            )

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.invert_yaxis()  # Bildkoordinaten: y zeigt nach unten
        ax.set_xlabel(
            "→/← Sample   ↑/↓ Buchstabe   D=loeschen   Q=beenden\n"
            + state["info"]
        )
        fig.canvas.draw_idle()

    def on_key(event):
        letter = ALPHABET[state["letter"]]
        files = files_for(letter)
        state["info"] = ""

        if event.key in ("right", "n"):
            state["i"] += 1
        elif event.key in ("left", "p"):
            state["i"] -= 1
        elif event.key == "up":
            state["letter"] = (state["letter"] + 1) % len(ALPHABET)
            state["i"] = 0
        elif event.key == "down":
            state["letter"] = (state["letter"] - 1) % len(ALPHABET)
            state["i"] = 0
        elif event.key in ("d", "delete") and files:
            path = files[state["i"] % len(files)]
            os.remove(path)
            state["info"] = f"GELOESCHT: {os.path.basename(path)}"
            print(f"[Review] geloescht: {path}")
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser("GestureRecognition-RecordLetters")
    # --mode muss existieren, damit die Engine kein None-Mode bekommt
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--label", action="store", default="A",
                        help="Buchstabe, mit dem gestartet wird (A-Z)")
    parser.add_argument("--review", action="store_true",
                        help="Aufnahmen durchschauen und loeschen statt aufnehmen")
    parser.add_argument("--webcam.width", required=False)

    # Modelldatei relativ zur Projekt-Wurzel finden, egal von wo gestartet wird
    os.chdir(PROJECT_ROOT)

    # Startlabel schon vor dem Engine-Parsing herausziehen
    known, _ = parser.parse_known_args()
    start_label = known.label.upper()

    if known.review:
        review_recordings(start_label)
        return

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
