#!/usr/bin/env python3
"""Create corrected DEAP copies using the supplied participant-ratings metadata."""
import argparse

from cfnn.labels import repair_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/data_preprocessed_python")
    parser.add_argument("--metadata", default="data/metadata_xls/participant_ratings.xls")
    parser.add_argument("--output-dir", default="data/corrected/data_preprocessed_python")
    args = parser.parse_args()
    repair_labels(args.data_dir, args.metadata, args.output_dir)


if __name__ == "__main__":
    main()
