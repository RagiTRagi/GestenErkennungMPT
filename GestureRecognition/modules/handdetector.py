import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, get_nested_key, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


def draw_hand_landmarks(hand_landmarks, galy: GALY, w, h):
    """Zeichnet die Landmarken und Verbindungen einer erkannten Hand.

    Die normalisierten MediaPipe-Koordinaten werden in Pixelkoordinaten
    umgerechnet. Finger und Handfläche erhalten jeweils eigene Farben; jeder
    Landmark-Punkt wird zusätzlich als Kreis markiert.

    Parameters
    ----------
    hand_landmarks : sequence
        MediaPipe-Landmarken einer Hand mit normalisierten ``x``- und
        ``y``-Koordinaten.
    galy : GALY
        Zeichenobjekt, dem Linien und Kreise hinzugefügt werden.
    w : int
        Breite des Kamerabildes in Pixeln.
    h : int
        Höhe des Kamerabildes in Pixeln.

    Returns
    -------
    None
        Die Zeichenbefehle werden direkt in ``galy`` eingetragen.
    """
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    x = np.inf
    y = np.inf
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (int(hand_landmarks[conn.start].x * w), 
                    int(hand_landmarks[conn.start].y * h))

            end = (int(hand_landmarks[conn.end].x * w), 
                int(hand_landmarks[conn.end].y * h))
            x = min(x, start[0], end[0])
            y = min(y, start[1], end[1])
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((int(hand_landmarks[pt].x * w), int(hand_landmarks[pt].y * h)), 5, (255,255,255), 1)
            galy.circle((int(hand_landmarks[pt].x * w), int(hand_landmarks[pt].y * h)), 4, lm[key]["color"], -1)


class HandDetector(Module):
    """Erkennt eine Hand in Webcam-Frames und visualisiert ihre Landmarken.

    Das Modul verwendet den MediaPipe Hand Landmarker im Bildmodus. Es nimmt
    die Signale ``config`` und ``webcam`` entgegen und gibt das
    Detektionsergebnis zusammen mit den zugehörigen GALY-Zeichenbefehlen aus.
    """

    def __init__(self, outputSignal="detector"):
        """Registriert die Ein- und Ausgabesignale des Detektormoduls.

        Parameters
        ----------
        outputSignal : str, optional
            Name des Signals im Ausgabeschema. Standardmäßig ``"detector"``.
        """
        super().__init__(
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )

    def start(self, data):
        """Lädt das MediaPipe-Modell und erstellt den Hand Landmarker.

        Der Modellpfad wird aus ``config["handdetector_model_path"]`` gelesen,
        sofern dieser Eintrag vorhanden ist. Andernfalls wird
        ``hand_landmarker.task`` verwendet. Der Detektor läuft im Bildmodus
        und erkennt höchstens eine Hand.

        Parameters
        ----------
        data : dict
            Framework-Daten. Das optionale ``config``-Dictionary kann den
            Pfad zur Modelldatei enthalten.

        Returns
        -------
        dict
            Leeres Dictionary, da bei der Initialisierung kein Signal erzeugt
            wird.
        """

        # Try to get model path from config, fallback to default
        model_path = "hand_landmarker.task"
        if "config" in data:
            config = data["config"]
            if isinstance(config, dict) and "handdetector_model_path" in config:
                model_path = config["handdetector_model_path"]

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        return {}

    def step(self, data):
        """Erkennt Handlandmarken in einem einzelnen Webcam-Frame.

        Das OpenCV-Bild wird von BGR nach RGB konvertiert und als
        :class:`mediapipe.Image` an den Detektor übergeben. Erkannte
        Landmarken und ihre Verbindungen werden in einem :class:`GALY`-Objekt
        eingezeichnet.

        Parameters
        ----------
        data : dict
            Framework-Daten mit ``webcam`` als BGR-Bild im Format
            ``(Höhe, Breite, Kanäle)``.

        Returns
        -------
        dict
            Bei vorhandenem Bild ein Dictionary mit dem MediaPipe-Ergebnis
            unter ``detector`` und der Visualisierung unter ``galy``. Ist das
            Webcam-Signal ``None``, wird ein leeres Dictionary zurückgegeben.
        """

        webcam = data["webcam"]
        if webcam is not None:
            h, w, _ = webcam.shape
            bild = cv2.cvtColor(webcam, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bild)

            result = self.detector.detect(mp_image)

            galy = GALY()

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    draw_hand_landmarks(hand_lms, galy, w, h)

            return {"detector": result, "galy": galy}

        return {}

    def stop(self, data):
        """Schließt den erzeugten MediaPipe-Detektor.

        Parameters
        ----------
        data : dict
            Letzte Framework-Daten. Sie werden von dieser Methode nicht
            verwendet.

        Returns
        -------
        None
        """
        if hasattr(self, 'detector'):
            self.detector.close()
