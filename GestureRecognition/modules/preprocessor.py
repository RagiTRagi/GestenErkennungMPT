from SignalHub import GALY, get_nested_key, Module
from collections import deque
import numpy as np

class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung von Fingertrajektorien.

    Sammelt die Position einer Finger-Landmarke aus dem ``detector``
    Signal über mehrere Frames hinweg, normalisiert die entstehende
    Trajektorie in Lage und Größe und stellt sie als Signal für
    nachfolgende Module bereit, z.B. für die HMM-Klassifikation.
    """

    def __init__(self, outputSignal="preprocessor"):
        """
        Registriert das Modul beim Framework.

        Abonniert werden ``config`` (Systemkonfiguration) und
        ``detector`` (Ergebnisse der Handdetektion). Erzeugt wird ein
        Signal, das entweder eine normalisierte Trajektorie oder
        ``None`` enthält.

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        """
        Liest die Parameter aus der Konfiguration und legt den
        Trajektorien-Puffer an.

        Wird einmal beim Start des Moduls ausgeführt. Gelesen werden
        ``preprocessor.finger_idx`` (Index der verfolgten Landmarke),
        ``preprocessor.max_lost`` (Anzahl Frames ohne Hand, nach denen
        die Trajektorie verworfen wird), ``preprocessor.min_steps``
        (Mindestanzahl Punkte für eine Ausgabe) und
        ``preprocessor.buffer_size`` (maximale Länge der Trajektorie).

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
        self.finger_idx = get_nested_key("preprocessor.finger_idx", config)
        self.max_lost = get_nested_key("preprocessor.max_lost", config)
        self.min_steps = get_nested_key("preprocessor.min_steps", config)
        buffer_size = get_nested_key("preprocessor.buffer_size", config)

        self.trajectory = deque(maxlen=buffer_size)
        self.lost_count = 0
        return {}

    def step(self, data):
        """
        Verarbeitet einen einzelnen Frame.

        Wurde eine Hand erkannt und der konfigurierte Finger ist
        ausgestreckt, wird die Landmarke ``finger_idx`` an die
        Trajektorie angehängt. Ohne erkannte Hand zählt ``lost_count``
        hoch; nach mehr als ``max_lost`` Frames gilt die Geste als
        abgebrochen und die Trajektorie wird geleert.

        Sobald mindestens ``min_steps`` Punkte vorliegen, wird die
        Trajektorie normalisiert: Der Mittelwert wird abgezogen, sodass
        sie im Ursprung zentriert ist, und anschließend durch den größten
        Absolutwert geteilt, sodass alle Koordinaten in ``[-1, 1]``
        liegen. Das Ergebnis ist damit unabhängig davon, wo im Bild und
        wie groß die Geste geschrieben wurde.

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            ``{outputSignal: pts}`` mit der normalisierten Trajektorie
            als :class:`numpy.ndarray` der Form ``(n, 2)``, oder
            ``{outputSignal: None}``, solange noch zu wenige Punkte
            gesammelt wurden.
        """
        detector_result = data.get("detector")

        if detector_result is not None and detector_result.handedness:
            landmarks = detector_result.hand_landmarks
            finger_landmark = landmarks[0][self.finger_idx]
            pip_landmark = landmarks[0][self.finger_idx - 2]

            if finger_landmark.y <= pip_landmark.y:
                self.trajectory.append((finger_landmark.x, finger_landmark.y))
                self.lost_count = 0
            else:
                self.lost_count += 1
                if self.lost_count > self.max_lost:
                    self.trajectory.clear()
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.trajectory.clear()

        if len(self.trajectory) < self.min_steps:
            return {self.outputSignal: None}

        pts = np.array(self.trajectory)
        pts = pts - pts.mean(axis=0)
        scale = np.abs(pts).max()
        if scale > 0:
            pts = pts / scale

        return {self.outputSignal: pts}

    def stop(self, data):
        """
        Wird beim Beenden des Moduls aufgerufen.

        Das Modul hält keine externen Ressourcen, daher ist keine
        Bereinigung nötig.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass
