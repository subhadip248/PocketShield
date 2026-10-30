"""Offline, rule-based checks for potentially risky URLs.

This module never visits a URL. Its indicators are heuristics, not a verdict:
legitimate URLs can trigger them, and phishing URLs can evade them.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import SplitResult, urlsplit, urlunsplit


# A small, explicit set of commonly used public URL-shortening services.
URL_SHORTENERS = frozenset(
    {
        "bit.ly",
        "buff.ly",
        "cutt.ly",
        "goo.gl",
        "is.gd",
        "ow.ly",
        "rebrand.ly",
        "t.co",
        "tinyurl.com",
        "shorturl.at",
    }
)

LONG_URL_LENGTH = 2000
MAX_SUBDOMAINS = 3

# A small offline list of common multi-label public suffixes. This is not a
# complete public suffix list; see _count_subdomains for the fallback behavior.
MULTI_LABEL_PUBLIC_SUFFIXES = frozenset(
    {
        "ac.uk", "co.in", "co.jp", "co.nz", "co.uk", "co.za",
        "com.au", "com.br", "com.cn", "com.hk", "com.mx", "com.my",
        "com.sg", "com.tr", "com.tw", "gov.uk", "net.au", "org.au",
        "org.uk",
    }
)


def _invalid_result(original_url: str, explanation: str) -> dict:
    """Return a consistent result when the input is empty or cannot be parsed."""
    return {
        "original_url": original_url,
        "normalized_url": None,
        "analysis_status": "invalid",
        # None/Unknown prevents a failed analysis being mistaken for low risk.
        "risk_score": None,
        "risk_level": "Unknown",
        "is_suspicious": None,
        "detected_indicators": [],
        "explanation": explanation,
    }


def _parse_url(url: str) -> tuple[SplitResult, str] | None:
    """Parse an absolute HTTP(S) URL and return its parts and hostname."""
    if not isinstance(url, str) or not url.strip():
        return None
    if any(char.isspace() or unicodedata.category(char) == "Cc" for char in url):
        return None

    try:
        parts = urlsplit(url.strip())
        hostname = parts.hostname
        # Accessing .port validates malformed and out-of-range port numbers.
        _ = parts.port
    except ValueError:
        return None

    if parts.scheme.lower() not in {"http", "https"} or not hostname:
        return None
    try:
        hostname.encode("idna")
    except UnicodeError:
        return None
    return parts, hostname


def _normalize_url(parts: SplitResult, hostname: str) -> str:
    """Lowercase the scheme and host while preserving path, query, and fragment."""
    host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literals need brackets in a URL authority.

    userinfo = ""
    if "@" in parts.netloc:
        userinfo = parts.netloc.rsplit("@", 1)[0] + "@"

    try:
        port = parts.port
    except ValueError:  # Normally caught during parsing; defensive for callers.
        port = None
    authority = f"{userinfo}{host}"
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parts.scheme.lower(), authority, parts.path or "/", parts.query, parts.fragment))


def _is_ip_address(hostname: str) -> bool:
    """Return whether a hostname is an IPv4 or IPv6 literal."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _count_subdomains(hostname: str) -> int:
    """Estimate subdomains using a small offline suffix list.

    For listed suffixes such as ``co.uk``, the registrable domain is the
    label immediately before the two-part suffix. Other domains use the
    common one-label suffix assumption. This cannot cover every public suffix.
    """
    normalized_host = hostname.rstrip(".").lower()
    labels = normalized_host.split(".")
    if len(labels) < 2:
        return 0

    matching_suffix = next(
        (
            suffix
            for suffix in sorted(MULTI_LABEL_PUBLIC_SUFFIXES, key=len, reverse=True)
            if normalized_host.endswith("." + suffix)
        ),
        None,
    )
    suffix_label_count = len(matching_suffix.split(".")) if matching_suffix else 1
    subdomain_count = max(0, len(labels) - suffix_label_count - 1)

    # Preserve the previous convention that a leading www is not counted.
    if labels[0] == "www" and subdomain_count:
        subdomain_count -= 1
    return subdomain_count


def _suspicious_character_reasons(url: str, parts: SplitResult, hostname: str) -> list[str]:
    """Find character and URL-shape cues often used to disguise destinations."""
    reasons: list[str] = []
    if "@" in parts.netloc:
        reasons.append("The URL contains user information before the host, which can disguise the actual destination.")
    if "\\" in url:
        reasons.append("The URL contains a backslash, which can be interpreted inconsistently by URL handlers.")
    if re.search(r"%0[0-9a-f]|%1[0-9a-f]|%7f", url, flags=re.IGNORECASE):
        reasons.append("The URL contains percent-encoded control characters.")
    if any(label.startswith("xn--") for label in hostname.lower().split(".")):
        reasons.append("The hostname contains an internationalized (punycode) label that may visually resemble another domain.")
    elif any(not label.isascii() for label in hostname.split(".")):
        reasons.append("The hostname contains Unicode characters that may visually resemble characters from another script.")
    return reasons


def _risk_level(score: int) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def analyze_url(url: str) -> dict:
    """Analyze a URL string locally and return heuristic risk indicators.

    The score is the capped sum of simple indicator weights, not a probability.
    HTTPS only avoids the plain-HTTP indicator; it does not establish that a
    site is legitimate or safe.
    """
    original = url if isinstance(url, str) else str(url)
    if not isinstance(url, str) or not url.strip():
        return _invalid_result(original, "No URL was provided; there is nothing to analyze.")

    parsed = _parse_url(url)
    if parsed is None:
        return _invalid_result(
            original,
            "The input is not a valid absolute HTTP or HTTPS URL, so URL indicators could not be assessed.",
        )

    parts, hostname = parsed
    normalized = _normalize_url(parts, hostname)
    indicators: list[str] = []
    score = 0

    if parts.scheme.lower() == "http":
        indicators.append("Uses HTTP without transport encryption.")
        score += 20

    if _is_ip_address(hostname):
        indicators.append("Uses an IP address directly instead of a domain name.")
        score += 30

    if hostname.lower().rstrip(".") in URL_SHORTENERS:
        indicators.append("Uses a recognized URL-shortening service, which hides the destination until followed.")
        score += 25

    if len(url) > LONG_URL_LENGTH:
        indicators.append(f"URL is unusually long (over {LONG_URL_LENGTH} characters).")
        score += 15

    subdomain_count = _count_subdomains(hostname)
    if subdomain_count > MAX_SUBDOMAINS:
        indicators.append(f"Hostname has {subdomain_count} subdomains, more than the threshold of {MAX_SUBDOMAINS}.")
        score += 15

    character_reasons = _suspicious_character_reasons(url, parts, hostname)
    if character_reasons:
        indicators.extend(character_reasons)
        score += 15 * len(character_reasons)

    score = min(score, 100)
    level = _risk_level(score)
    explanation = (
        "No configured risk indicators were found. This does not prove the site is safe."
        if not indicators
        else " ".join(indicators) + " These rules provide a heuristic warning, not a phishing verdict."
    )

    return {
        "original_url": original,
        "normalized_url": normalized,
        "analysis_status": "analyzed",
        "risk_score": score,
        "risk_level": level,
        "is_suspicious": score >= 30,
        "detected_indicators": indicators,
        "explanation": explanation,
    }
