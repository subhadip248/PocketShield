"""Offline, rule-based checks for potentially scam-like SMS messages.

The detector only examines the supplied text. It does not send messages,
visit links, or make network requests. Its score is a heuristic, not a verdict.
"""

from __future__ import annotations

import re

from pocketshield.phishing.url_detector import URL_SHORTENERS, analyze_url


RISK_THRESHOLD = 30
MAX_RISK_SCORE = 100

# One weight is added when each category is detected. We require context for
# credential/payment requests so words such as "OTP" or "bank" alone do not
# make a message suspicious.
INDICATOR_WEIGHTS = {
    "prize_or_lottery": 25,
    "actionable_prize_redemption": 10,
    "urgent_account_verification": 20,
    "suspicious_payment_request": 25,
    "credential_request": 35,
    "suspicious_link": 30,
    "threatening_or_urgent_language": 15,
    "paid_service_offer": 30,
    "suspicious_promotional_offer": 30,
    "paid_callback_lure": 30,
}

PRIZE_PATTERN = re.compile(
    r"\b(?:you(?:'ve| have)?\s+won|congratulations|winner|lottery|jackpot|"
    r"prize\s+money|claim\s+(?:your\s+)?(?:prize|reward)|"
    r"win.{0,35}(?:cash|prize|voucher|holiday|trip|£\s?\d)|"
    r"(?:selected|specially\s+selected).{0,45}(?:receive|for).{0,35}(?:prize|reward|cash|bonus|holiday|voucher)|"
    r"awarded.{0,35}(?:prize|reward|cash|bonus|holiday|voucher))\b",
    re.IGNORECASE,
)

ACCOUNT_CONTEXT_PATTERN = re.compile(r"\baccount\b", re.IGNORECASE)
ACCOUNT_ACTION_PATTERN = re.compile(
    r"\b(?:verify|verification|confirm|validate|locked|suspend(?:ed|sion)?|restricted)\b",
    re.IGNORECASE,
)

URGENT_PATTERN = re.compile(
    r"\b(?:urgent(?:ly)?|immediately|act\s+now|within\s+\d+\s+(?:minutes?|hours?|days?)|"
    r"final\s+warning|will\s+be\s+(?:closed|suspended|blocked)|legal\s+action|"
    r"arrest|penalty|expires?\s+(?:today|soon))\b",
    re.IGNORECASE,
)

_ACTION_WORD = r"(?:send|share|provide|reply(?:\s+with)?|text|enter|confirm|verify|forward|tell|give)"
_SENSITIVE_ITEM = (
    r"(?:otp|one[- ]time\s+(?:password|code)|pin|password|"
    r"bank(?:ing)?\s+(?:login|credentials|details)|card\s+(?:number|details)|cvv|security\s+code)"
)
CREDENTIAL_REQUEST_PATTERN = re.compile(
    rf"\b(?:{_ACTION_WORD}\b.{{0,45}}\b{_SENSITIVE_ITEM}\b|"
    rf"\b{_SENSITIVE_ITEM}\b.{{0,45}}\b{_ACTION_WORD}\b)",
    re.IGNORECASE,
)
SAFETY_WARNING_PATTERN = re.compile(
    rf"\b(?:never|do\s+not|don't|dont|avoid)\s+(?:{_ACTION_WORD})\b"
    rf".{{0,50}}\b{_SENSITIVE_ITEM}\b",
    re.IGNORECASE,
)

_ACTION_WORD_PAYMENT = r"(?:pay|send|transfer|wire|purchase|buy)"
_MONEY_TERM = r"(?:payment|money|fee|funds|cash|gift\s*cards?|bitcoin|crypto|usd|dollars?|\$\s?\d+)"
PAYMENT_REQUEST_PATTERN = re.compile(
    rf"\b(?:{_ACTION_WORD_PAYMENT}\b.{{0,40}}\b{_MONEY_TERM}\b|"
    rf"\b{_MONEY_TERM}\b.{{0,40}}\b{_ACTION_WORD_PAYMENT}\b)",
    re.IGNORECASE,
)

# URL matching is intentionally limited to ordinary web links and bare domain
# names. The existing URL analyzer then checks each candidate locally.
URL_PATTERN = re.compile(
    r"(?i)(?:https?://|www\.)[^\s<>\"']+|"
    r"(?<![@\w.-])(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>\"']*)?"
)
LINK_CALL_TO_ACTION_PATTERN = re.compile(
    r"\b(?:click|tap|visit|open|login|log\s+in|sign\s+in|verify|update|confirm|claim|track)\b",
    re.IGNORECASE,
)
TRAILING_URL_PUNCTUATION = ".,!?;:)]}"

# New commercial-spam signals require combinations of offer/service context,
# a transaction cue, and a response action. These remain deliberately narrow:
# words such as "free", "win", "call", or "subscription" are not enough.
PRIZE_ACTION_PATTERN = re.compile(
    r"\b(?:call|phone|text|txt|reply|send|visit|go\s+to)\b.{0,70}"
    r"\b(?:claim|redeem|collect|receive|enter|quote|quoting)\b|"
    r"\b(?:claim|redeem|collect|receive|enter|quote|quoting)\b.{0,70}"
    r"\b(?:call|phone|text|txt|reply|send|visit|go\s+to)\b",
    re.IGNORECASE,
)
PRIZE_REDEMPTION_DETAIL_PATTERN = re.compile(
    r"\b(?:claim\s+code|claim\s+number|quote\s+(?:the\s+)?(?:code|reference)|"
    r"text\s+[A-Z0-9]{2,}\s+to\s+\d{4,}|reply\s+[A-Z0-9]{2,}\s+to\s+\d{4,})\b",
    re.IGNORECASE,
)

PAID_RATE_PATTERN = re.compile(
    r"\b(?:premium[- ]rate|\d+(?:\.\d+)?\s*p\s*(?:/|per\s*)(?:msg|message|text|wk|week|day|minute|min)|"
    r"\d+(?:\.\d+)?\s*p(?:pm|/min)|\d+(?:\.\d+)?\s*(?:gbp|pounds?)\s+per\s+"
    r"(?:msg|message|text|week|month|day)|charged\s+at\s+\d+(?:\.\d+)?\s*p)\b|"
    r"£\s?\d+(?:\.\d+)?\s*(?:/|per\s*)(?:msg|message|text|week|month|day)",
    re.IGNORECASE,
)
PAID_SMS_ACTION_PATTERN = re.compile(
    r"\b(?:text|txt|send|reply)\s+(?:the\s+)?[A-Z0-9]{2,}\s+to\s+\d{4,}\b|"
    r"\b(?:text|txt|send|reply)\b.{0,50}\b(?:subscribe|join|enter|buy|purchase|activate|"
    r"start|continue|confirm|reply)\b",
    re.IGNORECASE,
)
SUBSCRIPTION_CONTEXT_PATTERN = re.compile(
    r"\b(?:ring ?tone|subscription|subscribe|weekly\s+service|content\s+service|"
    r"download\s+service|mobile\s+service)\b",
    re.IGNORECASE,
)
RECURRING_CHARGE_PATTERN = re.compile(
    r"\b(?:charged|charge|cost|costs|pay|paying)\b.{0,35}"
    r"(?:£\s?\d|\d+(?:\.\d+)?\s*(?:p|gbp|pounds?))?.{0,12}"
    r"\b(?:per\s+(?:week|month|day|message|msg|text)|/\s*(?:wk|week|month|day|msg|text)|"
    r"(?:\d+\s*)?(?:p/wk|p/week|p/month))\b|"
    r"(?:£\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:p|gbp|pounds?))\s*/\s*(?:wk|week|month|day)",
    re.IGNORECASE,
)
SUBSCRIPTION_ACTION_PATTERN = re.compile(
    r"\b(?:reply|text|txt|send)\s+(?:yes|no|stop|start|join|subscribe|continue|cancel)\b|"
    r"\b(?:activate|continue|confirm|renew|join|subscribe)\b",
    re.IGNORECASE,
)

OFFER_BENEFIT_PATTERN = re.compile(
    r"\b(?:free|complimentary|bonus|entitled|selected|specially\s+selected|"
    r"guaranteed|special\s+offer|awarded)\b",
    re.IGNORECASE,
)
OFFER_ITEM_PATTERN = re.compile(
    r"\b(?:mobile|phone|handset|camcorder|camera|voucher|holiday|trip|upgrade|"
    r"offer|gift|reward|cash|prize|credit|minutes?)\b",
    re.IGNORECASE,
)
OFFER_ACTION_PATTERN = re.compile(
    r"\b(?:call|phone|text|txt|reply|claim|redeem|collect|join|visit|click)\b",
    re.IGNORECASE,
)
OFFER_TARGETING_PATTERN = re.compile(
    r"\b(?:you(?:'re|\s+are)?\s+(?:selected|entitled|awarded)|selected\s+to\s+receive|"
    r"specially\s+selected|claim\s+(?:your|the)|offer\s+(?:expires|valid)|"
    r"call\s+now\s+to\s+claim)\b",
    re.IGNORECASE,
)

CALLBACK_CONTEXT_PATTERN = re.compile(
    r"\b(?:important\s+customer\s+service\s+announcement|customer\s+service\s+"
    r"(?:announcement|representative)|delivery\s+waiting|arrange\s+delivery|"
    r"tried\s+to\s+contact\s+you|we\s+tried\s+to\s+call\s+you|"
    r"your\s+(?:reward|prize|bonus|cash)\s+(?:is\s+)?waiting)\b",
    re.IGNORECASE,
)
CALLBACK_ACTION_PATTERN = re.compile(
    r"\b(?:call|phone)\b.{0,45}\b(?:now|today|to\s+claim|to\s+arrange|back|us|our)\b|"
    r"\b(?:call|phone)\s+(?:us|our|now)\b",
    re.IGNORECASE,
)
PREMIUM_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+44\s?)?(?:0?9\d{8,}|0?87[0-3]\d{6,}|0?70\d{8})(?!\d)")
PAID_SERVICE_WARNING_PATTERN = re.compile(
    r"\b(?:never|do\s+not|don't|dont|avoid|beware|block|report)\b.{0,75}"
    r"\b(?:text|txt|reply|respond|answer|send|call|dial|use|subscribe|premium[- ]rate|paid\s+(?:text|sms|service)|"
    r"short\s+code|\d{4,})\b|"
    r"\b(?:scam|fraud|warning|beware)\b.{0,75}"
    r"\b(?:premium[- ]rate|charged\s+at|\d+(?:\.\d+)?\s*p\s*(?:/|per)|"
    r"paid\s+(?:text|sms|service)|short\s+code)\b",
    re.IGNORECASE,
)


def _invalid_result(reason: str) -> dict:
    """Build a consistent result that cannot be mistaken for a safe analysis."""
    return {
        "analysis_status": "invalid",
        "risk_score": None,
        "risk_level": "Unknown",
        "is_suspicious": None,
        "reasons": [reason],
    }


def _has_account_verification_request(message: str) -> bool:
    """Require both an account reference and a verification or restriction cue."""
    return bool(ACCOUNT_CONTEXT_PATTERN.search(message) and ACCOUNT_ACTION_PATTERN.search(message))


def _has_credential_request(message: str) -> bool:
    """Find a request to provide a secret, excluding clear safety warnings."""
    if SAFETY_WARNING_PATTERN.search(message):
        return False
    return bool(CREDENTIAL_REQUEST_PATTERN.search(message))


def _has_suspicious_link(message: str) -> bool:
    """Check links with the existing offline URL detector; never open them."""
    for match in URL_PATTERN.finditer(message):
        candidate = match.group(0).rstrip(TRAILING_URL_PUNCTUATION)
        if not candidate:
            continue

        if candidate.lower().startswith("www."):
            candidate = "https://" + candidate
        elif not candidate.lower().startswith(("http://", "https://")):
            candidate = "https://" + candidate

        url_result = analyze_url(candidate)
        if url_result["analysis_status"] != "analyzed":
            continue

        # A shortener or a URL with strong configured URL indicators is risky.
        host = candidate.split("/", 3)[2].split(":", 1)[0].lower().rstrip(".")
        if host in URL_SHORTENERS or url_result["risk_score"] >= RISK_THRESHOLD:
            return True

    # A link used to send someone to a login, verification, or claim action is
    # suspicious in context, even if its hostname has no configured URL flags.
    return bool(URL_PATTERN.search(message) and LINK_CALL_TO_ACTION_PATTERN.search(message))


def _is_paid_service_safety_warning(message: str) -> bool:
    """Recognize advice warning people away from paid SMS/callback scams."""
    return bool(PAID_SERVICE_WARNING_PATTERN.search(message))


def _has_paid_service_offer(message: str) -> bool:
    """Require explicit price/rate and an action, or billed subscription context."""
    if PAID_RATE_PATTERN.search(message) and PAID_SMS_ACTION_PATTERN.search(message):
        return True
    return bool(
        SUBSCRIPTION_CONTEXT_PATTERN.search(message)
        and RECURRING_CHARGE_PATTERN.search(message)
        and SUBSCRIPTION_ACTION_PATTERN.search(message)
    )


def _has_suspicious_promotional_offer(message: str) -> bool:
    """Require a targeted benefit, identifiable offer item, and redemption action."""
    return bool(
        OFFER_BENEFIT_PATTERN.search(message)
        and OFFER_ITEM_PATTERN.search(message)
        and OFFER_ACTION_PATTERN.search(message)
        and OFFER_TARGETING_PATTERN.search(message)
    )


def _has_paid_callback_lure(message: str) -> bool:
    """Require a callback lure and a number with known premium-rate prefixes."""
    return bool(
        CALLBACK_CONTEXT_PATTERN.search(message)
        and CALLBACK_ACTION_PATTERN.search(message)
        and PREMIUM_PHONE_PATTERN.search(message)
    )


def _collect_indicators(message: str) -> list[tuple[str, str]]:
    """Return detected indicator keys and beginner-readable explanations."""
    indicators: list[tuple[str, str]] = []
    paid_service_warning = _is_paid_service_safety_warning(message)
    has_prize_claim = bool(PRIZE_PATTERN.search(message))
    has_paid_service_offer = not paid_service_warning and _has_paid_service_offer(message)
    has_promo_offer = (
        not paid_service_warning
        and not has_prize_claim
        and not has_paid_service_offer
        and _has_suspicious_promotional_offer(message)
    )

    if has_prize_claim and not paid_service_warning:
        indicators.append(("prize_or_lottery", "The message claims the recipient won a prize, lottery, or reward."))
        if PRIZE_ACTION_PATTERN.search(message) or PRIZE_REDEMPTION_DETAIL_PATTERN.search(message):
            indicators.append(
                ("actionable_prize_redemption", "The prize message directs the recipient to redeem it by phone, text, or a claim code.")
            )

    if _has_account_verification_request(message):
        indicators.append(
            ("urgent_account_verification", "The message links an account verification or restriction to an action.")
        )

    if PAYMENT_REQUEST_PATTERN.search(message):
        indicators.append(("suspicious_payment_request", "The message asks the recipient to send or pay money."))

    if _has_credential_request(message):
        indicators.append(
            ("credential_request", "The message asks the recipient to provide an OTP, PIN, password, or banking secret.")
        )

    if _has_suspicious_link(message):
        indicators.append(("suspicious_link", "The message contains a risky link or pairs a link with an action request."))

    if URGENT_PATTERN.search(message) and not paid_service_warning:
        indicators.append(("threatening_or_urgent_language", "The message uses urgency, a threat, or a deadline to pressure the recipient."))

    if has_paid_service_offer:
        indicators.append(
            ("paid_service_offer", "The message combines a paid text or subscription service with a response or enrollment action.")
        )
    elif has_promo_offer:
        indicators.append(
            ("suspicious_promotional_offer", "The message targets the recipient with a benefit and directs them to redeem an unsolicited offer.")
        )
    elif (
        not paid_service_warning
        and not has_prize_claim
        and _has_paid_callback_lure(message)
    ):
        indicators.append(
            ("paid_callback_lure", "The message pairs a callback lure with a known premium-rate phone number.")
        )

    return indicators


def analyze_sms(message: str) -> dict:
    """Analyze SMS text locally and return its heuristic risk assessment.

    Invalid input receives an ``invalid`` status and unknown risk values. For
    valid text, the score adds one weight per detected category, capped at 100.
    A score of 30 or more is marked suspicious. The score is not a probability.
    """
    if not isinstance(message, str):
        return _invalid_result("The message must be provided as text.")
    if not message.strip():
        return _invalid_result("The message is empty; there is nothing to analyze.")

    indicators = _collect_indicators(message)
    score = min(sum(INDICATOR_WEIGHTS[key] for key, _ in indicators), MAX_RISK_SCORE)
    if score >= 60:
        level = "High"
    elif score >= RISK_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    reasons = [reason for _, reason in indicators]
    if not reasons:
        reasons.append("No configured scam indicators were found; this does not prove the message is safe.")

    return {
        "analysis_status": "analyzed",
        "risk_score": score,
        "risk_level": level,
        "is_suspicious": score >= RISK_THRESHOLD,
        "reasons": reasons,
    }
