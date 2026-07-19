from SignalHub import Module, get_nested_key
from collections import deque
import numpy as np
from SignalHub import GALY

class TrailMarker(Module):
    """
    Verarbeitet Hand-Landmarks zur Erkennung und Speicherung einer Fingerspur.

    Das Modul empfängt Konfigurationsdaten sowie die Ergebnisse der
    Handerkennung. Die Position eines ausgewählten Finger-Landmarks wird über
    mehrere Frames hinweg verarbeitet, sodass eine zeitliche Bewegungsspur
    entsteht.

    Die erzeugte Spur kann zur Visualisierung der Fingerbewegung oder als
    Eingabesequenz für weitere Verarbeitungsschritte, beispielsweise eine
    Gestenerkennung, verwendet werden.

    Notes
    -----
    Das Modul erwartet die Eingangssignale ``"config"`` und ``"detector"``.
    Die konkrete Verarbeitung erfolgt in den Methoden ``start`` und ``step``.
    """

    def __init__(self, outputSignal="trailmarker"):
        """
        Initialisiert das TrailMarker-Modul.

        Der Konstruktor registriert das Modul bei der Basisklasse :class:`Module`.
        Dabei werden die benötigten Eingangssignale, das Output-Schema und der
        interne Modulname festgelegt.

        Parameters
        ----------
        outputSignal : str, optional
            Name des Ausgangssignals, unter dem die Ergebnisse des Moduls
            veröffentlicht werden. Der Standardwert ist ``"trailmarker"``.

        Notes
        -----
        Das Modul abonniert die Eingangssignale ``"config"`` und ``"detector"``.
        Das Output-Schema enthält eine Eigenschaft mit dem Namen des angegebenen
        Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisiert den internen Zustand des TrailMarker-Moduls.

        Die Methode wird einmal beim Start des Moduls aufgerufen. Sie liest die
        benötigten Parameter aus der Konfiguration und legt alle Zustandsvariablen
        an, die während der Verarbeitung einzelner Frames benötigt werden.

        Dazu gehören beispielsweise der zu verfolgende Finger-Landmark, die
        maximale Anzahl tolerierter Frames ohne erkannte Hand sowie eine begrenzte
        Historie vergangener Fingerpositionen. Diese Historie kann später zur
        Glättung, Visualisierung oder Weiterverarbeitung der Fingertrajektorie
        verwendet werden.

        Parameters
        ----------
        data : dict
            Vom Framework bereitgestellte Eingangsdaten. Das Dictionary enthält
            unter anderem das Signal ``"config"`` mit den benötigten
            Modulparametern.

        Returns
        -------
        dict
            Leeres Dictionary, da während der Initialisierung noch keine
            Ausgabedaten erzeugt werden.

        Notes
        -----
        Die Methode initialisiert ausschließlich Konfigurationswerte und interne
        Zustände. Die eigentliche Verarbeitung der Landmark-Daten erfolgt in der
        Methode ``step``.

        Für die Speicherung einer begrenzten Anzahl vergangener Fingerpositionen
        eignet sich eine :class:`collections.deque` mit festgelegter maximaler
        Länge.
        """

        config = data["config"]
        self.W = get_nested_key("width", config["webcam"])
        self.H = get_nested_key("height", config["webcam"])

        self.lost_frames = 0
        self.max_lostframe = 30
        self.finger_index = 8  # 8 = Index Finger
        self.distance_threshold = 2

        self.trajectory = deque(maxlen=2)
        self.final_trajectory = deque(maxlen=250)

        window_size = 3
        self.kernel_window = np.ones(window_size) / window_size
        return {}

    def draw_trajectory(self, galy):
        """
        Glättet die gespeicherte Fingertrajektorie und zeichnet sie als Linie.

        Die in ``self.final_trajectory`` gespeicherten Koordinaten werden in
        x- und y-Werte aufgeteilt und mithilfe des Glättungskernels
        ``self.kernel_window`` geglättet.

        Anschließend werden die geglätteten Punkte auf ganzzahlige
        Pixelkoordinaten gerundet und durch Linien miteinander verbunden. Die
        Trajektorie wird nur gezeichnet, wenn ausreichend viele Punkte für die
        Faltung vorhanden sind.

        Parameters
        ----------
        galy : object
            Zeichenobjekt mit einer ``line``-Methode, auf dem die geglättete
            Trajektorie dargestellt wird.

        Returns
        -------
        None
            Die Methode gibt keinen Wert zurück. Die Ausgabe wird direkt auf
            dem übergebenen Zeichenobjekt gezeichnet.

        Notes
        -----
        ``self.final_trajectory`` muss zweidimensionale Koordinaten enthalten.
        ``self.kernel_window`` wird als Faltungskern zur Glättung der
        Trajektorie verwendet.
        """
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
                prev_pt = tuple(np.round(smoothed_traj[i - 1]).astype(int))
                galy.line(curr_pt, prev_pt, (64, 224, 208))

    def step(self, data):
        """
        Verarbeitet einen Frame und aktualisiert die aufgezeichnete Fingerspur.

        Die Methode liest die erkannten Hand-Landmarks aus dem Eingangssignal
        ``"detector"``. Wird keine Hand erkannt, erhöht sie einen Zähler für
        verlorene Frames. Nach Überschreiten der erlaubten Anzahl werden die
        gespeicherten Trajektorien zurückgesetzt.

        Bei erkannter Hand wird geprüft, ob der ausgewählte Finger ausgestreckt
        ist. Nur dann wird seine normalisierte Landmark-Position in
        Pixelkoordinaten umgerechnet und zur Trajektorie hinzugefügt. Die bisher
        gespeicherte Fingerspur wird mithilfe von ``draw_trajectory`` auf einem
        GALY-Objekt dargestellt.

        Parameters
        ----------
        data : dict
            Eingangsdaten des Frameworks. Das Dictionary muss das Signal
            ``"detector"`` mit den erkannten Hand-Landmarks enthalten.

        Returns
        -------
        dict
            Ausgabedaten des aktuellen Frames. Das Dictionary enthält das
            GALY-Zeichenobjekt unter ``"galy"``. Wenn ein neuer Punkt
            aufgezeichnet wurde, enthält es zusätzlich dessen Koordinaten unter
            ``"trailmarker"``.

        Notes
        -----
        Wird für mehrere aufeinanderfolgende Frames keine Hand erkannt, werden
        ``self.trajectory`` und ``self.final_trajectory`` zurückgesetzt.

        Die Position des Fingers wird anhand von ``self.W`` und ``self.H`` in
        Bildkoordinaten umgerechnet und auf die gültigen Bildgrenzen begrenzt.

        Die Fingerprüfung vergleicht die Fingerspitze mit dem zugehörigen
        PIP-Gelenk und berücksichtigt zusätzlich den konfigurierten
        Distanzschwellenwert.
        """
        galy = GALY()
        detector = data["detector"]

        if detector is None:
            return {"galy": galy}

        landmarks = detector.hand_landmarks  # Landmarks pro Frame

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

        finger_landmark = landmarks[0][
            self.finger_index
        ]  # Landmarken des Fingers extrahieren
        pip = landmarks[0][self.finger_index - 2]
        finger_distance = np.sqrt((pip.y - finger_landmark.y)**2)
        extended_finger = (finger_landmark.y < pip.y and finger_distance<self.distance_threshold)
        
        if not extended_finger: 
            self.draw_trajectory(galy)
            return {"galy": galy}

        px = np.clip(finger_landmark.x * self.W, 0, self.W - 1)
        py = np.clip(finger_landmark.y * self.H, 0, self.H - 1)
        pt = (px, py)
        self.trajectory.append(pt)

        if len(self.trajectory) == 1:
            self.final_trajectory.append(self.trajectory[0])
            return {"trailmarker": pt, "galy": galy}

        current_pt = self.trajectory[-1]

        self.final_trajectory.append(current_pt)
        self.draw_trajectory(galy)
        return {"trailmarker": current_pt, "galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Parameters 
        ---------- 
        data : dict 
            Zuletzt vom Framework bereitgestellte Eingangsdaten. Die Daten 
            werden in der aktuellen Implementierung nicht verwendet. 
        
        Returns 
        ------- 
        dict 
            Leeres Dictionary, da beim Beenden des Moduls keine weiteren 
            Ausgabedaten erzeugt werden.
        """
        return {}