import argparse
import csv
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Official raisin.zip from UCI")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with ZipFile(args.archive) as outer:
        inner = outer.read("Raisin_Dataset.zip")
    with ZipFile(BytesIO(inner)) as source:
        arff = source.read("Raisin_Dataset/Raisin_Dataset.arff").decode("utf-8")
    header, rows = re.split(r"(?im)^@data\s*$", arff, maxsplit=1)
    columns = re.findall(r"(?im)^@attribute\s+(\S+)", header)
    values = list(csv.reader(line for line in rows.splitlines() if line.strip()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(values)
    print(len(values), "rows,", len(columns) - 1, "features")


if __name__ == "__main__":
    main()
