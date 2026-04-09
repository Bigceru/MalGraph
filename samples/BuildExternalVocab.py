import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


def iter_json_objects(path: Path) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    else:
        raise ValueError(f"Unsupported JSON content in: {path}")


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix in {".json", ".jsonl"}:
            return [input_path]
        raise ValueError(f"Unsupported input file format: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix in {".json", ".jsonl"}
    )
    return files


def count_external_functions(input_path: Path) -> Counter:
    files = list(iter_input_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found in: {input_path}")

    freq = Counter()

    for file in tqdm(files, desc="Scanning files"):
        for item in iter_json_objects(file):
            function_names = item.get("function_names", [])
            acfg_list = item.get("acfg_list", [])

            local_count = len(acfg_list)
            if local_count > len(function_names):
                raise ValueError(
                    f"Invalid sample in {file}: len(acfg_list) > len(function_names)"
                )

            external_names = function_names[local_count:]
            freq.update(external_names)

    return freq


def write_vocab_jsonl(freq: Counter, output_file: Path, min_freq: int = 1) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(
        ((name, count) for name, count in freq.items() if count >= min_freq),
        key=lambda x: (-x[1], x[0]),
    )

    # If the vocabulary already exists, increment counts instead of overwriting
    if output_file.exists():
        existing_freq = Counter()
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_freq[obj["f_name"]] += obj["count"]
                except (json.JSONDecodeError, KeyError):
                    continue

        # Merge existing frequencies with new frequencies
        for name, count in sorted_items:
            existing_freq[name] += count

        # Re-sort after merging
        sorted_items = sorted(
            ((name, count) for name, count in existing_freq.items() if count >= min_freq),
            key=lambda x: (-x[1], x[0]),
        )

    with output_file.open("w", encoding="utf-8") as f:
        for name, count in sorted_items:
            f.write(json.dumps({"f_name": name, "count": count}) + "\n")


class ExternalVocabBuilder:
    """Build and save external-function vocabulary frequencies.

    This class allows using the vocabulary generation logic programmatically
    from other Python modules.
    """

    def __init__(self, input_path: str, output_file: str = "./train_external_function_name_vocab.jsonl", min_freq: int = 1):
        self.input_path = Path(input_path)
        self.output_file = Path(output_file)
        self.min_freq = min_freq

    def count(self) -> Counter:
        return count_external_functions(self.input_path)

    def save(self, freq: Counter) -> None:
        write_vocab_jsonl(freq=freq, output_file=self.output_file, min_freq=self.min_freq)

    def run(self) -> Counter:
        freq = self.count()
        self.save(freq)
        return freq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build external function vocabulary frequency file for MalGraph."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .json/.jsonl file or directory containing them.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./train_external_function_name_vocab.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Keep only function names with count >= min_freq.",
    )

    args = parser.parse_args()

    builder = ExternalVocabBuilder(
        input_path=args.input,
        output_file=args.output,
        min_freq=args.min_freq,
    )
    freq = builder.run()

    print(f"Total unique external functions: {len(freq)}")
    print(f"Saved vocabulary frequencies to: {builder.output_file}")


if __name__ == "__main__":
    main()