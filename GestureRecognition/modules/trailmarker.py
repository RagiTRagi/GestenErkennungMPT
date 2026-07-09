from SignalHub import Module, get_nested_key
from collections import deque
import numpy as np
from SignalHub import GALY
from SignalHub.mode import EngineMode
import msvcrt

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

<<<<<<< HEAD

        self.finger_index = 8 # 8 = Index
        self.trajectory = deque(maxlen=2)
        self.lost_frames = 0
        self.final_trajectory = deque(maxlen=250)
        self.max_lostframe = 30
        #self.key = cv2.waitKey(1)
        window_size = 5
=======
        self.lost_frames = 0
        self.max_lostframe = 30
        self.finger_index = 8 # 8 = Index Finger

        self.trajectory = deque(maxlen=2)
        self.final_trajectory = deque(maxlen=250)
        
        window_size = 3
>>>>>>> origin/main
        self.kernel_window = np.ones(window_size) / window_size
        return {}
    
    def draw_trajectory(self, galy):
<<<<<<< HEAD
       smoothed_traj = np.convolve(self.final_trajectory, self.kernel_window, mode="valid")
       for i in range(len(smoothed_traj)):
            if i == 0:
              continue
            curr_pt = self.final_trajectory[i]
            prev_pt = self.final_trajectory[i-1]
            galy.line(curr_pt, prev_pt, (64, 224, 208))
       
=======
        
      if len(self.final_trajectory) > len(self.kernel_window):
        traj = np.array(self.final_trajectory)
        x = traj[:, 0]
        y = traj[:, 1]

        smoothed_x = np.convolve(x, self.kernel_window, mode="valid")
        smoothed_y = np.convolve(y, self.kernel_window, mode="valid")
        smoothed_traj = list(zip(smoothed_x, smoothed_y))

        for i in range(len(smoothed_traj)):
            if i == 0:
              continue
            curr_pt = tuple(np.round(smoothed_traj[i]).astype(int))
            prev_pt = tuple(np.round(smoothed_traj[i-1]).astype(int))
            galy.line(curr_pt, prev_pt, (64, 224, 208))


        
        
        
>>>>>>> origin/main

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

<<<<<<< HEAD
            ``return { ..., "galy": galy}``
        """
        if msvcrt.kbhit():
          if msvcrt.getch()  == b'q':
            return ({}, EngineMode.TERMINATE)
          
        galy = GALY()
        detector = data["detector"]

        if detector is None:
          return {"galy": galy}
        
        landmarks = detector.hand_landmarks # Landmarks pro Frame

        
        if landmarks is None or len(landmarks) == 0:
          self.lost_frames += 1

          if self.lost_frames >= self.max_lostframe:
            self.final_trajectory.clear()
            self.trajectory.clear()
            self.lost_frames = 0
            return {"galy": galy}
          
          self.draw_trajectory(galy)
          return {"galy": galy}
        
        self.lost_frames = 0

        finger_landmark = landmarks[0][self.finger_index] # Landmarken des Fingers extrahieren
        pip = landmarks[0][self.finger_index-2]
        extended_finger = finger_landmark.y < pip.y

        if not extended_finger:
           self.draw_trajectory(galy)
           return {"galy": galy}
        
        px = np.clip(finger_landmark.x*self.W, 0, self.W-1)
        py = np.clip(finger_landmark.y*self.H, 0, self.H-1)
        pt = (px, py)
        self.trajectory.append(pt)

        if len(self.trajectory) == 1:
          self.final_trajectory.append(self.trajectory[0])
          return {"trailmarker": pt, "galy": galy}

        previous_pt = self.trajectory[-2]
        current_pt = self.trajectory[-1]

        d = np.sqrt((previous_pt[0]-current_pt[0])**2 + (previous_pt[1]-current_pt[1])**2)

        if d >= 60.0 or d < 1.0:
          self.trajectory.pop()
          self.draw_trajectory(galy)
          return {"galy": galy}
        
        self.final_trajectory.append(current_pt)
        self.draw_trajectory(galy)
        return {"trailmarker": current_pt, "galy": galy}
=======
          ``return { ..., "galy": galy}``
      """
      if msvcrt.kbhit():
        if msvcrt.getch()  == b'q':
          return ({}, EngineMode.TERMINATE)
        
      galy = GALY()
      detector = data["detector"]

      if detector is None:
        return {"galy": galy}
      
      landmarks = detector.hand_landmarks # Landmarks pro Frame

      
      if landmarks is None or len(landmarks) == 0:
        self.lost_frames += 1

        if self.lost_frames >= self.max_lostframe:
          self.final_trajectory.clear()
          self.trajectory.clear()
          self.lost_frames = 0
          return {"galy": galy}
        
        self.draw_trajectory(galy)
        return {"galy": galy}
      
      self.lost_frames = 0

      finger_landmark = landmarks[0][self.finger_index] # Landmarken des Fingers extrahieren
      pip = landmarks[0][self.finger_index-2]
      extended_finger = finger_landmark.y < pip.y

      if not extended_finger:
          self.draw_trajectory(galy)
          return {"galy": galy}
      
      px = np.clip(finger_landmark.x*self.W, 0, self.W-1)
      py = np.clip(finger_landmark.y*self.H, 0, self.H-1)
      pt = (px, py)
      self.trajectory.append(pt)

      if len(self.trajectory) == 1:
        self.final_trajectory.append(self.trajectory[0])
        return {"trailmarker": pt, "galy": galy}

      previous_pt = self.trajectory[-2]
      current_pt = self.trajectory[-1]

      d = np.sqrt((previous_pt[0]-current_pt[0])**2 + (previous_pt[1]-current_pt[1])**2)

      if d >= 60.0 or d < 1.0:
        self.trajectory.pop()
        self.draw_trajectory(galy)
        return {"galy": galy}
      
      self.final_trajectory.append(current_pt)
      self.draw_trajectory(galy)
      return {"trailmarker": current_pt, "galy": galy}
>>>>>>> origin/main

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

<<<<<<< HEAD
        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        return {}
        #return {"trailmarker": self.final_trajectory.copy()}
    
=======
      Parameters
      ----------
      data : dict
          Letzte übergebene Daten des Frameworks.
      """
      return {}
>>>>>>> origin/main
