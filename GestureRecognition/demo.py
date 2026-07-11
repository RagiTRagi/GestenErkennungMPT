from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules import *
import argparse

def run(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--recorder.file", action="store")
    parser.add_argument("--engine.singlestep", action="store_true", default=False)
    parser.add_argument("--webcam.width", required=False)
    trailmarker = TrailMarker()
    preprocessor = Preprocessor()
    hmm = HMMModule()
    modules = [
        ConfigParser(parser),
        Webcam(),
        HandDetector(),
        trailmarker,
        preprocessor,
        hmm,
        # Taste R = alles zuruecksetzen (Spur, Puffer, Prognose)
        ResetButton(trailmarker=trailmarker, preprocessor=preprocessor, hmm=hmm),
    ]
    engine = Engine(modules=modules, signals={})
    signals = engine.run({})

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GestureRecognition")
    run(parser)