from SignalHub import Module, get_nested_key
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
        self.galy = GALY()
        self.galy.canvas("Trajectory", shape=(self.H, self.W), color=(0, 0, 0))
        self.finger_index = 1 # Category(index=1, score=0.5351641178131104, display_name='Left', category_name='Left')]] so wird finger bestimmt ?
        self.trajectory = deque(maxlen=100)
        self.lost_frames = 0
        self.distance_threshold = (10, 10)
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

        # TO-DO
        # - Punktdaten korrekt zurücknormalisieren (semi geschafft)
        # - dafür sorgen, dass bei lost frames die trajektorie zeichnung bleibt ✅
        # - linien aufeinanderaufbauen lassen ✅
        # - shape mit nested aus config extrahieren bezüglich frame größe
        # - Finger Index Wahl 
        # --------------------------------------
        
        
        landmarks = data["detector"]
        landmarks = landmarks.hand_landmarks # Landmarks pro Frame
        #print("Ddaten:", landmarks)
        if len(landmarks) == 0:
            self.lost_frames += 1
            #print(self.lost_frames)
            return {"galy": self.galy}
        
        
        for landmark in landmarks[0]:
          ptn = (landmark.x, landmark.y)
          pt = (np.float32(landmark.x*255), np.float32(landmark.y*255))
          
          self.trajectory.append(pt)

        #print(self.lost_frames)
        #print("Landmarks:", landmarks)
        
        for i in range(len(self.trajectory)-1):
          if len(self.trajectory) <= 1:
            return {}
          
          current_pt = self.trajectory[i]
          before_pt = self.trajectory[i-1]
          d = np.sqrt((before_pt[0]-current_pt[0])**2 + (before_pt[1]))
          #print("Points: ", (current_pt, next_pt))
          if difference > self.distance_threshold:
            print(difference)
            return {}
          
          self.galy.circle(current_pt, 1, (0, 102, 204), thickness=2)
        
          return {"galy": self.galy}

          # HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
          # Daten: HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
          #
        #return {}

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
        pass
    