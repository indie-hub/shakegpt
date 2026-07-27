#!/usr/bin/env python3
"""Prepare the local Project Gutenberg Shakespeare corpus deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "TrainingData"
SOURCE = DATA / "pg100.txt"
MARKER = "<|endoftext|>"

# Exact collection/work headings observed in pg100.txt, in source order.
WORKS = (
    ("THE SONNETS", "sonnets", "poetry"),
    ("ALL’S WELL THAT ENDS WELL", "alls-well-that-ends-well", "comedy"),
    ("THE TRAGEDY OF ANTONY AND CLEOPATRA", "antony-and-cleopatra", "tragedy"),
    ("AS YOU LIKE IT", "as-you-like-it", "comedy"),
    ("THE COMEDY OF ERRORS", "comedy-of-errors", "comedy"),
    ("THE TRAGEDY OF CORIOLANUS", "coriolanus", "tragedy"),
    ("CYMBELINE", "cymbeline", "tragedy"),
    ("THE TRAGEDY OF HAMLET, PRINCE OF DENMARK", "hamlet", "tragedy"),
    ("THE FIRST PART OF KING HENRY THE FOURTH", "henry-iv-part-1", "history"),
    ("THE SECOND PART OF KING HENRY THE FOURTH", "henry-iv-part-2", "history"),
    ("THE LIFE OF KING HENRY THE FIFTH", "henry-v", "history"),
    ("THE FIRST PART OF HENRY THE SIXTH", "henry-vi-part-1", "history"),
    ("THE SECOND PART OF KING HENRY THE SIXTH", "henry-vi-part-2", "history"),
    ("THE THIRD PART OF KING HENRY THE SIXTH", "henry-vi-part-3", "history"),
    ("KING HENRY THE EIGHTH", "henry-viii", "history"),
    ("THE LIFE AND DEATH OF KING JOHN", "king-john", "history"),
    ("THE TRAGEDY OF JULIUS CAESAR", "julius-caesar", "tragedy"),
    ("THE TRAGEDY OF KING LEAR", "king-lear", "tragedy"),
    ("LOVE’S LABOUR’S LOST", "loves-labours-lost", "comedy"),
    ("THE TRAGEDY OF MACBETH", "macbeth", "tragedy"),
    ("MEASURE FOR MEASURE", "measure-for-measure", "comedy"),
    ("THE MERCHANT OF VENICE", "merchant-of-venice", "comedy"),
    ("THE MERRY WIVES OF WINDSOR", "merry-wives-of-windsor", "comedy"),
    ("A MIDSUMMER NIGHT’S DREAM", "midsummer-nights-dream", "comedy"),
    ("MUCH ADO ABOUT NOTHING", "much-ado-about-nothing", "comedy"),
    ("THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE", "othello", "tragedy"),
    ("PERICLES, PRINCE OF TYRE", "pericles", "comedy"),
    (
        "THE LIFE AND DEATH OF KING RICHARD THE SECOND",
        "richard-ii",
        "history",
    ),
    ("KING RICHARD THE THIRD", "richard-iii", "history"),
    ("THE TRAGEDY OF ROMEO AND JULIET", "romeo-and-juliet", "tragedy"),
    ("THE TAMING OF THE SHREW", "taming-of-the-shrew", "comedy"),
    ("THE TEMPEST", "tempest", "comedy"),
    ("THE LIFE OF TIMON OF ATHENS", "timon-of-athens", "tragedy"),
    ("THE TRAGEDY OF TITUS ANDRONICUS", "titus-andronicus", "tragedy"),
    ("TROILUS AND CRESSIDA", "troilus-and-cressida", "tragedy"),
    ("TWELFTH NIGHT; OR, WHAT YOU WILL", "twelfth-night", "comedy"),
    ("THE TWO GENTLEMEN OF VERONA", "two-gentlemen-of-verona", "comedy"),
    ("THE TWO NOBLE KINSMEN", "two-noble-kinsmen", "comedy"),
    ("THE WINTER’S TALE", "winters-tale", "comedy"),
    ("A LOVER’S COMPLAINT", "lovers-complaint", "poetry"),
    ("THE PASSIONATE PILGRIM", "passionate-pilgrim", "poetry"),
    ("THE PHOENIX AND THE TURTLE", "phoenix-and-the-turtle", "poetry"),
    ("THE RAPE OF LUCRECE", "rape-of-lucrece", "poetry"),
    ("VENUS AND ADONIS", "venus-and-adonis", "poetry"),
)

# One complete work from each major category, selected as the four-work
# combination closest to 10% of cleaned bytes in this source.
VALIDATION_TITLES = {
    "THE TRAGEDY OF KING LEAR",
    "THE WINTER’S TALE",
    "KING RICHARD THE THIRD",
    "VENUS AND ADONIS",
}

# Fourteen complete sonnets spread evenly through the 154-poem sequence.
VALIDATION_SONNETS = {
    6,
    17,
    28,
    39,
    50,
    61,
    72,
    83,
    94,
    105,
    116,
    127,
    138,
    149,
}

FORBIDDEN_GUTENBERG_TEXT = (
    "project gutenberg",
    "gutenberg license",
    "*** start of",
    "*** end of",
    "www.gutenberg.org",
)


def fail(message: str) -> None:
    raise ValueError(message)


def heading_positions(lines: list[str]) -> list[int]:
    positions = []
    for title, _, _ in WORKS:
        matches = []
        for index, line in enumerate(lines):
            separated = index == 0 or (
                index >= 3
                and all(not value.strip() for value in lines[index - 3 : index])
                and sum(
                    not value.strip() for value in lines[index + 1 : index + 5]
                )
                >= 2
            )
            if line == title and separated:
                matches.append(index)
        if len(matches) != 1:
            fail(f"expected one structural heading for {title!r}, found {matches}")
        positions.append(matches[0])
    if positions != sorted(positions) or positions[0] != 0:
        fail("work headings are missing, out of order, or preceded by unknown text")
    return positions


def clean_lines(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        line = unicodedata.normalize("NFC", line.rstrip())
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


def extract_sonnets(lines: list[str]) -> list[dict[str, object]]:
    if not lines or lines[0] != "THE SONNETS":
        fail("malformed Sonnets collection boundary")

    headings = []
    for index, line in enumerate(lines):
        number = line.strip()
        if (
            number.isdigit()
            and 1 <= int(number) <= 154
            and index > 0
            and not lines[index - 1].strip()
            and index + 1 < len(lines)
            and not lines[index + 1].strip()
        ):
            headings.append((int(number), index))
    if [number for number, _ in headings] != list(range(1, 155)):
        fail("expected one ordered heading for each of the 154 sonnets")

    endings = [
        index
        for index, line in enumerate(lines[headings[-1][1] :], headings[-1][1])
        if line.strip() == "THE END"
    ]
    if len(endings) != 1:
        fail(f"expected one Sonnets end boundary, found {endings}")

    sonnets = []
    for offset, (number, start) in enumerate(headings):
        end = headings[offset + 1][1] if offset + 1 < len(headings) else endings[0]
        text = clean_lines(lines[start:end])
        if text.splitlines()[0].strip() != str(number) or len(text) < 100:
            fail(f"implausible extraction for Sonnet {number}")
        encoded = text.encode("utf-8")
        sonnets.append(
            {
                "title": f"Sonnet {number}",
                "filename": f"works/sonnet-{number:03}.txt",
                "category": "poetry",
                "split": "validation" if number in VALIDATION_SONNETS else "train",
                "utf8_bytes": len(encoded),
                "characters": len(text),
                "text": text,
            }
        )
    return sonnets


def extract_works(source_bytes: bytes) -> list[dict[str, object]]:
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        fail(f"source is not valid UTF-8: {error}")

    source_text = unicodedata.normalize(
        "NFC", source_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = source_text.split("\n")
    positions = heading_positions(lines)
    if next((line.strip() for line in reversed(lines) if line.strip()), None) != "FINIS":
        fail("source does not end with the expected FINIS boundary")

    extracted = []
    for number, ((title, slug, category), start) in enumerate(
        zip(WORKS, positions)
    ):
        end = positions[number + 1] if number + 1 < len(positions) else len(lines)
        work_lines = lines[start:end]

        if title == "THE SONNETS":
            extracted.extend(extract_sonnets(work_lines))
            continue

        if category != "poetry":
            contents = [
                index
                for index, line in enumerate(work_lines[:250])
                if line.strip() == "Contents"
            ]
            personae = [
                index
                for index, line in enumerate(work_lines[:350])
                if line.strip() in {"Dramatis Personæ", "Dramatis Personae"}
            ]
            if (
                len(contents) != 1
                or len(personae) != 1
                or contents[0] >= personae[0]
            ):
                fail(
                    f"malformed contents/personae boundary in {title!r}: "
                    f"contents={contents}, personae={personae}"
                )
            work_lines = work_lines[: contents[0]] + work_lines[personae[0] :]

        text = clean_lines(work_lines)
        if not text.startswith(title + "\n") or len(text) < 1_000:
            fail(f"implausible extraction for {title!r}")
        if MARKER in text:
            fail(f"reserved marker occurs in {title!r}")
        if any(fragment in text.casefold() for fragment in FORBIDDEN_GUTENBERG_TEXT):
            fail(f"Project Gutenberg wrapper text remains in {title!r}")
        if "\n\n\n" in text:
            fail(f"excessive blank lines remain in {title!r}")

        encoded = text.encode("utf-8")
        extracted.append(
            {
                "title": title,
                "filename": f"works/{slug}.txt",
                "category": category,
                "split": "validation" if title in VALIDATION_TITLES else "train",
                "utf8_bytes": len(encoded),
                "characters": len(text),
                "text": text,
            }
        )
    return extracted


def build_outputs(source_bytes: bytes) -> tuple[dict[Path, bytes], dict[str, object]]:
    works = extract_works(source_bytes)
    combined = {
        split: "".join(
            str(work["text"]) + MARKER + "\n"
            for work in works
            if work["split"] == split
        )
        for split in ("train", "validation")
    }
    work_bytes = {
        split: sum(
            int(work["utf8_bytes"]) for work in works if work["split"] == split
        )
        for split in ("train", "validation")
    }
    total_work_bytes = sum(work_bytes.values())

    manifest_works = [
        {key: value for key, value in work.items() if key != "text"} for work in works
    ]
    manifest: dict[str, object] = {
        "source": "TrainingData/pg100.txt",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "separator": MARKER,
        "validation_selection": {
            "method": (
                "Fixed complete-work selection plus 14 complete sonnets distributed "
                "evenly through the 154-poem sequence."
            ),
            "titles": sorted(VALIDATION_TITLES),
            "sonnet_numbers": sorted(VALIDATION_SONNETS),
        },
        "stats": {
            "work_count": len(works),
            "train_work_count": sum(work["split"] == "train" for work in works),
            "validation_work_count": sum(
                work["split"] == "validation" for work in works
            ),
            "train_sonnet_count": 154 - len(VALIDATION_SONNETS),
            "validation_sonnet_count": len(VALIDATION_SONNETS),
            "train_work_bytes": work_bytes["train"],
            "validation_work_bytes": work_bytes["validation"],
            "train_combined_bytes": len(combined["train"].encode("utf-8")),
            "validation_combined_bytes": len(
                combined["validation"].encode("utf-8")
            ),
            "validation_percentage_by_work_bytes": (
                work_bytes["validation"] / total_work_bytes * 100
            ),
        },
        "works": manifest_works,
    }

    outputs = {
        Path(str(work["filename"])): str(work["text"]).encode("utf-8")
        for work in works
    }
    outputs[Path("train.txt")] = combined["train"].encode("utf-8")
    outputs[Path("validation.txt")] = combined["validation"].encode("utf-8")
    outputs[Path("manifest.json")] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return outputs, manifest


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def write_outputs(outputs: dict[Path, bytes]) -> None:
    expected_works = {
        DATA / relative for relative in outputs if relative.parent == Path("works")
    }
    stale_works = set((DATA / "works").glob("*.txt")) - expected_works
    existing = [DATA / relative for relative in outputs if (DATA / relative).exists()]

    if existing:
        top_level = sum(path.parent == DATA for path in existing)
        work_files = len(existing) - top_level
        print(
            "Replacing generated artifacts: "
            f"{top_level} top-level file(s), {work_files} work file(s)."
        )
    if stale_works:
        print("Removing stale generated works:")
        for path in sorted(stale_works):
            print(f"  {path.relative_to(ROOT)}")

    for relative, data in outputs.items():
        atomic_write(DATA / relative, data)
    for path in stale_works:
        path.unlink()


def verify(
    outputs: dict[Path, bytes],
    manifest: dict[str, object],
    source_sha256_before: str,
) -> None:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_sha256_before:
        fail("original source changed during preparation")

    for relative, expected in outputs.items():
        path = DATA / relative
        if not path.is_file() or not path.stat().st_size:
            fail(f"missing or empty output: {path.relative_to(ROOT)}")
        actual = path.read_bytes()
        actual.decode("utf-8")
        if actual != expected:
            fail(f"non-deterministic or stale output: {path.relative_to(ROOT)}")

    entries = list(manifest["works"])
    train_titles = {entry["title"] for entry in entries if entry["split"] == "train"}
    validation_titles = {
        entry["title"] for entry in entries if entry["split"] == "validation"
    }
    if train_titles & validation_titles or len(train_titles | validation_titles) != len(
        entries
    ):
        fail("a work is duplicated or assigned to both splits")

    work_texts = {}
    for entry in entries:
        text = (DATA / str(entry["filename"])).read_text(encoding="utf-8")
        if MARKER in text:
            fail(f"marker found in {entry['filename']}")
        if len(text.encode("utf-8")) != entry["utf8_bytes"]:
            fail(f"byte count mismatch in {entry['filename']}")
        if len(text) != entry["characters"]:
            fail(f"character count mismatch in {entry['filename']}")
        work_texts[str(entry["title"])] = text

    for split in ("train", "validation"):
        assigned = [entry for entry in entries if entry["split"] == split]
        combined = (DATA / f"{split}.txt").read_text(encoding="utf-8")
        expected = "".join(
            work_texts[str(entry["title"])] + MARKER + "\n" for entry in assigned
        )
        if combined != expected:
            fail(f"{split}.txt has a missing or unmarked work boundary")
        if combined.count(MARKER) != len(assigned):
            fail(f"wrong marker count in {split}.txt")
        # This small corpus keeps the direct check readable. A much larger corpus
        # could precompute hashes instead of repeatedly scanning the combined text.
        for entry in entries:
            expected_count = int(entry["split"] == split)
            if combined.count(work_texts[str(entry["title"])]) != expected_count:
                fail(f"{entry['title']!r} does not occur exactly once in its split")
        if any(
            fragment in combined.casefold()
            for fragment in FORBIDDEN_GUTENBERG_TEXT
        ):
            fail(f"Project Gutenberg wrapper text remains in {split}.txt")

    actual_works = set((DATA / "works").glob("*.txt"))
    expected_works = {DATA / str(entry["filename"]) for entry in entries}
    if actual_works != expected_works:
        fail("works directory contains missing or unmanifested text files")


def print_summary(manifest: dict[str, object]) -> None:
    stats = manifest["stats"]
    print(f"Extracted works: {stats['work_count']}")
    print(f"Training works: {stats['train_work_count']}")
    print(f"Validation works: {stats['validation_work_count']}")
    print(f"Training sonnets: {stats['train_sonnet_count']}")
    print(f"Validation sonnets: {stats['validation_sonnet_count']}")
    print(f"Training bytes: {stats['train_combined_bytes']}")
    print(f"Validation bytes: {stats['validation_combined_bytes']}")
    print(
        "Validation percentage: "
        f"{stats['validation_percentage_by_work_bytes']:.4f}%"
    )
    print(
        "End-of-text markers: "
        f"train={stats['train_work_count']}, "
        f"validation={stats['validation_work_count']}"
    )
    print("Verification: passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="rebuild expected bytes in memory and verify existing outputs",
    )
    args = parser.parse_args()

    source_bytes = SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    outputs, manifest = build_outputs(source_bytes)
    if not args.verify_only:
        write_outputs(outputs)
    verify(outputs, manifest, source_sha256)
    print_summary(manifest)


if __name__ == "__main__":
    main()
