

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pocketshield.sms.sms_detector import analyze_sms  # noqa: E402


SPAM_THRESHOLD = 30
EXAMPLES_PER_REPORT = 20

# These are descriptive labels for this report only. They do not feed into or
# alter PocketShield's detector.
CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Prize, competition, or reward claims",
        re.compile(r"\b(?:win|won|winner|prize|award(?:ed)?|quiz|draw|jackpot|voucher|bonus|selected|claim)\b", re.I),
    ),
    (
        "Adult, dating, or chat services",
        re.compile(r"\b(?:sexy|horny|naked|dating|date|flirt|chat|babes?|dogging|laid|hardcore|blind\s+date)\b", re.I),
    ),
    (
        "Mobile phones or product promotions",
        re.compile(r"\b(?:mobile|nokia|handset|camcorder|camera\s+phone|colour\s+mobiles?|upgrade|upgrade|latest\s+phone)\b", re.I),
    ),
    (
        "Subscriptions, downloads, and media content",
        re.compile(r"\b(?:ringtone|tones|download|subscription|subscribe|weekly|horoscope|singles\s+chart|content|club|sports?\s+news)\b", re.I),
    ),
    (
        "Premium-rate calls or paid text services",
        re.compile(r"\b(?:\d+\s*(?:p\s*/\s*(?:msg|wk|week|day)|ppm|pence)|premium\s+rate|charged\s+at|cost\s+\d|\d{5,})\b|£\s?\d", re.I),
    ),
    (
        "Callbacks, delivery, or customer-service lures",
        re.compile(r"\b(?:call\s+(?:now|us|our)|call\s+back|tried\s+to\s+contact|customer\s+service|delivery\s+waiting|arrange\s+delivery|missed\s+call)\b", re.I),
    ),
)
OTHER_CATEGORY = "Other promotional or unsolicited messages"

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?", re.I)
NUMBER_RE = re.compile(r"(?<!\d)\+?\d(?:[\s()./-]*\d){6,}(?!\d)")
UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
SELF_NAME_RE = re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I)


@dataclass(frozen=True)
class MissedMessage:
    line_number: int
    message: str
    score: int
    category: str


def categorize(message: str) -> str:
    """Assign one descriptive primary pattern to a missed message."""
    for category, pattern in CATEGORIES:
        if pattern.search(message):
            return category
    return OTHER_CATEGORY


def mask_personal_details(message: str) -> str:
    """Mask common direct identifiers before printing example SMS text."""
    masked = EMAIL_RE.sub("[EMAIL]", message)
    masked = URL_RE.sub("[LINK]", masked)
    masked = NUMBER_RE.sub(
        lambda match: "[PHONE/NUMBER]" if sum(char.isdigit() for char in match.group()) >= 7 else match.group(),
        masked,
    )
    masked = UK_POSTCODE_RE.sub("[POSTCODE]", masked)
    masked = SELF_NAME_RE.sub(lambda match: match.group(0).split()[0] + " [NAME]", masked)
    masked = re.sub(r"(?<=-\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?=\s+-\s*(?:MSG|message))", "[NAME]", masked)
    return masked


def find_false_negatives(path: Path) -> tuple[list[MissedMessage], int]:
    missed: list[MissedMessage] = []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as dataset:
        for line_number, raw_line in enumerate(dataset, start=1):
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                malformed += 1
                continue
            label, message = line.split("\t", 1)
            if label.strip().lower() != "spam":
                continue
            result = analyze_sms(message)
            score = result.get("risk_score")
            if result.get("analysis_status") == "analyzed" and isinstance(score, (int, float)) and score < SPAM_THRESHOLD:
                missed.append(MissedMessage(line_number, message, int(score), categorize(message)))
    return missed, malformed


def print_report(missed: list[MissedMessage], malformed: int = 0) -> None:
    print("PocketShield false-negative SMS analysis")
    print(f"Spam predicted as ham (risk_score < {SPAM_THRESHOLD}): {len(missed)}")
    if malformed:
        print(f"Malformed rows skipped: {malformed}")

    print("\nPrimary pattern groups (exclusive counts):")
    counts = Counter(item.category for item in missed)
    for category, count in counts.most_common():
        print(f"  {category}: {count} ({count / len(missed):.1%})")
    score_counts = Counter(item.score for item in missed)
    print("\nDetector score among misses:")
    for score, count in sorted(score_counts.items()):
        print(f"  {score}: {count}")

    print(f"\n{min(EXAMPLES_PER_REPORT, len(missed))} representative missed SMS examples (identifiers masked):")
    
    grouped: dict[str, list[MissedMessage]] = defaultdict(list)
    for item in missed:
        grouped[item.category].append(item)
    chosen: list[MissedMessage] = []
    category_order = [name for name, _ in counts.most_common()]
    while len(chosen) < min(EXAMPLES_PER_REPORT, len(missed)):
        added = False
        for category in category_order:
            if grouped[category]:
                chosen.append(grouped[category].pop(0))
                added = True
                if len(chosen) == EXAMPLES_PER_REPORT:
                    break
        if not added:
            break
    for number, item in enumerate(chosen, start=1):
        print(f"  {number}. [{item.category}; score {item.score}; dataset row {item.line_number}] {mask_personal_details(item.message)}")

    print("\nWhy these messages were missed")
    print("  The SMS detector only scores its configured patterns. Many promotion, subscription, adult/chat, and service-lure messages do not match those patterns, so they receive 0 points. The configured prize/lottery pattern does match some broader prize wording, but its weight is 25; with no second matching signal, those messages remain below the 30-point cutoff. Urgency is recognized only for a fixed set of words and phrases, while account-verification, credential-request, payment-request, and suspicious-link signals require specific wording or context that many of these promotions lack.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        print("Place the UCI SMS Spam Collection file at data/raw/SMSSpamCollection or pass its path.", file=sys.stderr)
        return 1
    try:
        missed, malformed = find_false_negatives(args.dataset)
    except OSError as exc:
        print(f"Unable to read dataset {args.dataset}: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset: {args.dataset}")
    print_report(missed, malformed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
