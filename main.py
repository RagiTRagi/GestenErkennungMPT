import argparse

from GestureRecognition import run
from GestureRecognition.labeling import data_labeling

def main():
    """Startet wahlweise die Aufnahme oder die Gestenerkennung."""
    parser = argparse.ArgumentParser("GestureRecognition")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Startet die interaktive Aufnahme und Beschriftung von Gesten.",
    )

    args, _ = parser.parse_known_args()
    if args.record:
        data_labeling()
        return

    run(parser)

if __name__ == "__main__":
    main()
