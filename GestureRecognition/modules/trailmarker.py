from SignalHub import Module, bgr, get_nested_key
from collections import deque
import numpy as np
from SignalHub import GALY

class TrailMarker(Module):
    """
    Modul zum Zeichnen einer Spur anhand der Bewegung eines Fingers.

    Die Position eines bestimmten Finger-Landmarks wird über mehrere Frames
    hinweg gespeichert. Aus diesen Punkten kann anschließend eine Linie
    erzeugt werden, die den Bewegungsverlauf des Fingers visualisiert.

    Ziel ist es, die Verarbeitung der Landmark-Daten sowie die Verwaltung
    eines Zustands über mehrere Frames hinweg selbst zu implementieren.
    """

    def __init__(self, outputSignal="trailmarker"):
        """
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``detector`` : Ergebnisse der Handdetektion

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Da dieses Modul keine eigenen Daten erzeugt, reicht beispielsweise:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle Variablen vorzubereiten, die während der
        Laufzeit des Moduls benötigt werden.

        Hinweise
        --------
        - Lese benötigte Parameter aus der Konfiguration.
        - Bestimme beispielsweise, welcher Finger verfolgt werden soll.
        - Lege eine Datenstruktur an, in der mehrere vergangene
          Fingerpositionen gespeichert werden können,
          z.B. :class:`collections.deque` mit einer maximalen Größe.
        - Diese Historie wird später verwendet, um eine Spur zu zeichnen.
        - Speichere aus der Konfiguration weitere benötigte Parameter,
          z.B. Finger-Index, maximale Anzahl verlorener Frames oder
          Webcam-Parameter.
        - Für den Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
           Eine ``deque`` ist ideal für Trajektorien,
           da sie effizient alte Punkte entfernt.

        .. note::
           Initialisiere hier nur Zustände und Parameter,
           keine eigentliche Verarbeitung.

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks. Enthält unter anderem das
            Signal ``config``.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        
        config = data["config"]
        self.W = get_nested_key("width", config["webcam"])
        self.H = get_nested_key("height", config["webcam"])

        self.finger_index = 8 # 8 = Index
        self.trajectory = deque(maxlen=10)
        self.lost_frames = 0
        self.final_trajectory = []
        # persistente Speicherung von gezeichneten Strokes (Liste von Punktlisten)
        self.strokes = []
        # aktuell geführter Stroke (wird fortlaufend befüllt)
        self.current_stroke = []
        # Anzahl Frames, nach denen ein Stroke als beendet gilt
        self.stroke_finish_threshold = 3
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, die aktuelle Position eines Fingers zu bestimmen,
        diese Position in einer Trajektorie zu speichern und daraus
        eine visuelle Spur zu erzeugen.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte Hände und
          deren Landmarken zu erhalten.
        - Falls keine Hand erkannt wurde, kann beispielsweise ein Zähler
          für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position kann zur bestehenden Trajektorie hinzugefügt werden.
        - Zwischen aufeinanderfolgenden Punkten können Linien gezeichnet
          werden, um eine Spur darzustellen.
        - Für die Visualisierung kann :meth:`line` der :class:`GALY`
          verwendet werden.

        .. tip::
          Typischer Ablauf:
           1. Landmark extrahieren
           2. Punkt speichern
           3. Trajektorie aktualisieren
           4. Linien zwischen Punkten zeichnen

        .. warning::
            Achte darauf, dass:
              - keine leeren Landmark-Daten verarbeitet werden
              - die Trajektorie nicht unendlich wächst
              - verlorene Frames sinnvoll behandelt werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Um die Zeichenoperationen auszuführen, sollte ein
            :class:`GALY` Objekt zurückgegeben werden.

            Beispiel:

            ``return { ..., "galy": galy}``
        """

        galy = GALY()
        landmarks = data["detector"]
        #print("1",landmarks.hand_world_landmarks)
        landmarks = landmarks.hand_landmarks # Landmarks pro Frame
        #galy = data["galy"]
        #print("Ddaten:", landmarks)
        if len(landmarks) == 0:
          # keine Hand erkannt: erhöhten lost counter
          self.lost_frames += 1
          # falls ein aktueller Stroke existiert und wir ihn beenden sollten,
          # verschiebe ihn in die persistenten Strokes
          if self.current_stroke and self.lost_frames >= self.stroke_finish_threshold:
              self.strokes.append(self.current_stroke[:])
              self.current_stroke.clear()
          # redraw all stored strokes so the drawing persists
          for stroke in self.strokes:
              for i in range(1, len(stroke)):
                  galy.line(stroke[i-1], stroke[i], (0, 0, 0))
          for i in range(1, len(self.current_stroke)):
              galy.line(self.current_stroke[i-1], self.current_stroke[i], (0, 0, 0))
          return {"galy": galy}
        #print("2", landmarks[0][0])

        # Hand erkannt -> reset lost counter
        self.lost_frames = 0

        finger_landmark = landmarks[0][self.finger_index] # Landmarken des Fingers extrahieren
        pt = (finger_landmark.x * self.W, finger_landmark.y * self.H)

        # füge Punkt in die kurzfristige Trajektorie und in den aktuellen Stroke
        self.trajectory.append(pt)
        # wenn current_stroke leer, starte neuen Stroke
        if not self.current_stroke:
          self.current_stroke.append(pt)
        else:
          # Abstand zur letzten Position
          last = self.current_stroke[-1]
          d = float(np.hypot(last[0]-pt[0], last[1]-pt[1]))
          if d >= 70.0:
            # großer Sprung: beende aktuellen Stroke und starte neuen
            self.strokes.append(self.current_stroke[:])
            self.current_stroke = [pt]
          else:
            self.current_stroke.append(pt)

        # zeichne alle persistente Strokes neu (so bleiben sie sichtbar)
        for stroke in self.strokes:
          for i in range(1, len(stroke)):
            galy.line(stroke[i-1], stroke[i], (bgr("#FFE600")))
        # zeichne aktuellen Stroke
        for i in range(1, len(self.current_stroke)):
          galy.line(self.current_stroke[i-1], self.current_stroke[i], (bgr("#FFE600")))

        # optional: liefere komplette Sammlung zurück
        all_strokes = self.strokes + ([self.current_stroke] if self.current_stroke else [])
        return {"trailmarker": all_strokes, "galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber sinnvoll sein,
           wenn Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        print("last trajectory given")
        return {"trailmarker": self.final_trajectory}
    