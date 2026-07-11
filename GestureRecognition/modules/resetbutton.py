import ctypes
import os

from SignalHub import GALY, Module

# --- Windows key polling (wie in labelrecorder.py) -----------------------------
_user32 = ctypes.windll.user32

VK_R = 0x52


def _key_down(vk):
    return (_user32.GetAsyncKeyState(vk) & 0x8000) != 0


def _our_window_is_foreground():
    """Nur reagieren, wenn *unser* Fenster den Fokus hat."""
    hwnd = _user32.GetForegroundWindow()
    pid = ctypes.c_ulong()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value == os.getpid()


class ResetButton(Module):
    """
    Reset fuer die Live-Demo.

    Taste ``R`` (Kamera-Fenster fokussiert) setzt alles zurueck:
      - gesammelte Trajektorie des Preprocessors (Puffer)
      - sichtbare Spur des TrailMarkers
      - angezeigte Prognose des HMMModuls

    So kann man nach einem Buchstaben sauber mit dem naechsten anfangen,
    ohne dass alte Punkte im rollenden Fenster haengen bleiben.
    """

    def __init__(self, trailmarker=None, preprocessor=None, hmm=None):
        super().__init__(
            inputSignals=["config"],
            outputSchema={"type": "object", "properties": {"resetbutton": {}}},
            name="resetbutton",
        )
        self.trailmarker = trailmarker
        self.preprocessor = preprocessor
        self.hmm = hmm

    def start(self, data):
        self._prev = False
        self._flash = 0
        return {}

    def step(self, data):
        down = _key_down(VK_R) and _our_window_is_foreground()
        if down and not self._prev:
            self._reset()
        self._prev = down

        # kurze RESET-Einblendung nach dem Druecken
        if self._flash > 0:
            self._flash -= 1
            galy = GALY()
            galy.putText("RESET", (20, 80), fontScale=0.8,
                         color=(0, 200, 255), thickness=2)
            return {"galy": galy}

        return {}

    def _reset(self):
        if self.preprocessor is not None:
            self.preprocessor.trajectory.clear()
            self.preprocessor.lost_count = 0
        if self.trailmarker is not None:
            self.trailmarker.trajectory.clear()
            self.trailmarker.final_trajectory.clear()
            self.trailmarker.lost_frames = 0
        if self.hmm is not None:
            self.hmm.last_text = None
        self._flash = 20
        print("[ResetButton] Reset")

    def stop(self, data):
        return {}
