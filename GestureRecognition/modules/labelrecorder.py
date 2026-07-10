import ctypes
import os
import pickle

import numpy as np

from SignalHub import GALY, Module, get_nested_key

# --- Windows key polling (works regardless of Qt focus handling) --------------
_user32 = ctypes.windll.user32

VK_BACK = 0x08
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_S = 0x53
VK_R = 0x52


def _key_down(vk):
    return (_user32.GetAsyncKeyState(vk) & 0x8000) != 0


def _our_window_is_foreground():
    """Only react to hotkeys while *our* window has focus."""
    hwnd = _user32.GetForegroundWindow()
    pid = ctypes.c_ulong()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value == os.getpid()


class LabelRecorder(Module):
    """
    Persistentes Aufnahme-Modul fuer Buchstaben-Gesten.

    Statt fuer jede Aufnahme einen neuen Prozess zu starten (Webcam + MediaPipe
    + Fenster jedes Mal neu laden), haengt sich dieses Modul in die laufende
    SignalHub-Pipeline ein. Damit koennen beliebig viele Samples in *einer*
    Session aufgenommen werden.

    Steuerung (Fenster muss fokussiert sein):
      - ``S``            : aktuelle Trajektorie als Sample fuer den aktuellen
                           Buchstaben speichern und Puffer zuruecksetzen
      - ``Pfeil rechts`` : naechster Buchstabe (A -> B -> ... -> Z -> A)
      - ``Pfeil links``  : vorheriger Buchstabe
      - ``Backspace``    : letztes gespeichertes Sample wieder loeschen
      - ``Esc``          : Fenster schliessen / Session beenden

    Das gespeicherte Format ist kompatibel zu ``dataset_building`` in
    ``labeling.py``: ein Dict ``{"preprocessor": [{"preprocessor": <ndarray>}]}``.
    """

    ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self, preprocessor, recordings_dir, start_label="A"):
        super().__init__(
            inputSignals=["config", "preprocessor", "detector"],
            outputSchema={"type": "object", "properties": {"labeling": {}}},
            name="labelrecorder",
        )
        self.preprocessor = preprocessor
        self.recordings_dir = recordings_dir

        start_label = (start_label or "A").upper()
        self.idx = self.ALPHABET.index(start_label) if start_label in self.ALPHABET else 0

        self.min_steps = 15
        self.height = 360

        self._prev_key = {}
        self._saved_paths = []      # Reihenfolge aller in dieser Session gespeicherten Dateien
        self._flash = None          # (text, verbleibende_frames, color)

    # --- lifecycle ------------------------------------------------------------
    def start(self, data):
        config = data.get("config", {})
        self.min_steps = get_nested_key("preprocessor.min_steps", config) or self.min_steps
        self.height = get_nested_key("webcam.height", config) or self.height
        os.makedirs(self._label_dir(), exist_ok=True)
        print(f"[LabelRecorder] Aufnahme-Ordner: {self.recordings_dir}")
        print(f"[LabelRecorder] Start-Label: {self.label}   (Samples: {self._count()})")
        print("[LabelRecorder] S=speichern  <-/->=Buchstabe  Backspace=undo  Esc=Ende")
        return {}

    def step(self, data):
        self._handle_keys(data)

        traj = data.get("preprocessor")
        ready = isinstance(traj, np.ndarray) and len(traj) >= self.min_steps

        return {"galy": self._draw_hud(ready)}

    def stop(self, data):
        return {}

    # --- helpers --------------------------------------------------------------
    @property
    def label(self):
        return self.ALPHABET[self.idx]

    def _label_dir(self):
        return os.path.join(self.recordings_dir, self.label)

    def _count(self):
        d = self._label_dir()
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.endswith(".pickle")])

    def _switch(self, delta):
        self.idx = (self.idx + delta) % len(self.ALPHABET)
        os.makedirs(self._label_dir(), exist_ok=True)
        self._flash = (f"-> {self.label}", 12, (0, 255, 0))

    def _save(self, traj):
        if not (isinstance(traj, np.ndarray) and len(traj) >= self.min_steps):
            self._flash = ("Keine Trajektorie", 15, (0, 165, 255))
            return

        os.makedirs(self._label_dir(), exist_ok=True)
        r1 = np.random.randint(10000, 100000000)
        r2 = np.random.randint(1000, 1000000)
        filename = f"{self.label}_{r1}_{r2}.pickle"
        path = os.path.join(self._label_dir(), filename)

        sample = {"preprocessor": [{"preprocessor": np.array(traj, dtype=np.float64)}]}
        with open(path, "wb") as f:
            pickle.dump(sample, f)

        self._saved_paths.append(path)
        # Puffer leeren, damit der naechste Buchstabe sauber startet
        self.preprocessor.trajectory.clear()
        self.preprocessor.lost_count = 0
        self._flash = (f"SAVED {self.label} #{self._count()}", 15, (0, 255, 0))
        print(f"[LabelRecorder] gespeichert: {path}")

    def _undo(self):
        while self._saved_paths:
            path = self._saved_paths.pop()
            if os.path.exists(path):
                os.remove(path)
                self._flash = (f"UNDO {os.path.basename(path)}", 15, (60, 60, 255))
                print(f"[LabelRecorder] geloescht: {path}")
                return
        self._flash = ("Nichts zum Loeschen", 15, (0, 165, 255))

    def _handle_keys(self, data):
        if not _our_window_is_foreground():
            # Zustand trotzdem aktualisieren, damit kein "Nachfeuern" passiert
            for vk in (VK_S, VK_R, VK_LEFT, VK_RIGHT, VK_BACK):
                self._prev_key[vk] = _key_down(vk)
            return

        def pressed(vk):
            down = _key_down(vk)
            edge = down and not self._prev_key.get(vk, False)
            self._prev_key[vk] = down
            return edge

        save = pressed(VK_S) or pressed(VK_R)
        left = pressed(VK_LEFT)
        right = pressed(VK_RIGHT)
        undo = pressed(VK_BACK)

        if right:
            self._switch(+1)
        if left:
            self._switch(-1)
        if undo:
            self._undo()
        if save:
            self._save(data.get("preprocessor"))

    def _draw_hud(self, ready):
        galy = GALY()
        galy.layer("labeling", alwaysVisible=True)

        galy.putText(self.label, (15, 48), fontScale=1.5, color=(0, 255, 0), thickness=3)
        galy.putText(f"samples: {self._count()}", (95, 42), fontScale=0.7,
                     color=(0, 255, 0), thickness=2)

        if ready:
            galy.putText("READY - press S", (15, 82), fontScale=0.6,
                         color=(0, 255, 0), thickness=2)
        else:
            galy.putText("draw a letter...", (15, 82), fontScale=0.6,
                         color=(0, 200, 255), thickness=2)

        if self._flash is not None:
            text, frames, color = self._flash
            galy.putText(text, (15, 118), fontScale=0.8, color=color, thickness=2)
            frames -= 1
            self._flash = (text, frames, color) if frames > 0 else None

        galy.putText("S=save   <-/->=letter   Back=undo   Esc=quit",
                     (15, int(self.height) - 12), fontScale=0.5,
                     color=(255, 255, 255), thickness=1)
        return galy
