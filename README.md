# PocketShield

PocketShield is a cybersecurity project for the **Arm Bharat AI-SoC Challenge, Problem Statement 3: On-device Threat Detection**. Its goal is to provide fully offline cybersecurity assistance that analyzes information locally on a device, with the final deployment target of an **ARM64 Android smartphone**.

## Current status

The initial working module is an offline, rule-based phishing URL analyzer. It parses URL text locally and does not visit links or make network requests. SMS detection, network monitoring, and Llama integration are future modules and have not been implemented.

## Phishing URL detector features

The `pocketshield.phishing.url_detector.analyze_url(url)` function returns a structured result with analysis status, normalized URL, a 0–100 heuristic risk score, risk level, indicators, and an explanation. It currently checks for:

- Plain HTTP (without HTTPS).
- IP-address hostnames.
- A small documented list of URL-shortening services.
- URLs longer than 2,000 characters.
- More than three estimated subdomains.
- Suspicious URL patterns, including user information before the host, backslashes, encoded control characters, and Unicode or punycode hostnames.
- Invalid or empty URL input, reported with an `invalid` analysis status and an unknown risk level.

The implementation uses Python's standard library and runs without cloud services, AI models, or URL fetching.

## Development setup

From the project root, create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies listed for this project:

```bash
python -m pip install -r requirements.txt
```

Run the complete test suite from the project root:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

## Limitations

- The detector is rule-based heuristic software, not a guarantee of phishing detection or safety. HTTPS does not prove a site is trustworthy.
- Rules can produce false positives (flagging legitimate links) and false negatives (missing phishing links).
- The offline multi-label public suffix list is intentionally incomplete. Subdomain estimates may be inaccurate for suffixes that are not in its built-in list.
- URL parsing and normalization do not verify that a domain exists, who owns it, or what content it serves. URLs are never opened by this module.
