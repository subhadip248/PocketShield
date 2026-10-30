"""Tests for PocketShield's offline URL heuristic analyzer."""

import pytest

from pocketshield.phishing.url_detector import analyze_url


def test_normal_https_url_has_no_configured_indicators():
    result = analyze_url("https://www.example.com/account")

    assert result["normalized_url"] == "https://www.example.com/account"
    assert result["analysis_status"] == "analyzed"
    assert result["risk_score"] == 0
    assert result["risk_level"] == "Low"
    assert result["is_suspicious"] is False
    assert result["detected_indicators"] == []


def test_http_url_is_flagged():
    result = analyze_url("http://example.com/login")

    assert result["risk_score"] == 20
    assert any("HTTP" in indicator for indicator in result["detected_indicators"])


def test_ip_address_host_is_flagged():
    result = analyze_url("https://192.0.2.10/login")

    assert result["risk_score"] == 30
    assert result["is_suspicious"] is True
    assert any("IP address" in indicator for indicator in result["detected_indicators"])


def test_known_shortener_is_flagged():
    result = analyze_url("https://bit.ly/abc123")

    assert any("shortening service" in indicator for indicator in result["detected_indicators"])


def test_unusually_long_url_is_flagged():
    result = analyze_url("https://example.com/" + "a" * 2050)

    assert any("unusually long" in indicator for indicator in result["detected_indicators"])


def test_excessive_subdomains_are_flagged():
    result = analyze_url("https://a.b.c.d.e.example.com/login")

    assert any("subdomains" in indicator for indicator in result["detected_indicators"])


def test_invalid_url_returns_structured_result():
    result = analyze_url("not a URL")

    assert result["normalized_url"] is None
    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None
    assert result["risk_level"] == "Unknown"
    assert result["is_suspicious"] is None
    assert result["detected_indicators"] == []
    assert "not a valid" in result["explanation"]


def test_empty_input_returns_structured_result():
    result = analyze_url("")

    assert result["original_url"] == ""
    assert result["normalized_url"] is None
    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None
    assert result["risk_level"] == "Unknown"
    assert result["is_suspicious"] is None
    assert "No URL" in result["explanation"]


def test_multiple_indicators_accumulate_and_cap_at_100():
    result = analyze_url("http://user@192.0.2.10/" + "a" * 2050)

    assert result["risk_score"] == 80
    assert result["risk_level"] == "High"
    assert result["is_suspicious"] is True
    assert len(result["detected_indicators"]) >= 3


def test_ipv6_hostname_is_normalized_and_detected_as_ip():
    result = analyze_url("https://[2001:db8::1]:8443/login")

    assert result["analysis_status"] == "analyzed"
    assert result["normalized_url"] == "https://[2001:db8::1]:8443/login"
    assert any("IP address" in indicator for indicator in result["detected_indicators"])


def test_valid_port_is_preserved():
    result = analyze_url("https://example.com:8443/login")

    assert result["analysis_status"] == "analyzed"
    assert result["normalized_url"] == "https://example.com:8443/login"


def test_invalid_port_returns_invalid_status():
    result = analyze_url("https://example.com:99999/login")

    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None
    assert result["risk_level"] == "Unknown"


def test_trailing_dot_is_removed_from_normalized_hostname():
    result = analyze_url("https://example.com./login")

    assert result["analysis_status"] == "analyzed"
    assert result["normalized_url"] == "https://example.com/login"


def test_user_information_is_flagged_but_host_is_parsed_correctly():
    result = analyze_url("https://user:pass@example.com/login")

    assert result["analysis_status"] == "analyzed"
    assert "example.com" in result["normalized_url"]
    assert any("user information" in indicator for indicator in result["detected_indicators"])


@pytest.mark.parametrize(
    "url",
    [
        "https://раypal.com/login",  # Cyrillic letters mixed with Latin letters.
        "https://xn--ypal-43d9g.com/login",  # Punycode form of the mixed-script label.
    ],
)
def test_unicode_or_punycode_hostname_is_flagged(url):
    result = analyze_url(url)

    assert result["analysis_status"] == "analyzed"
    assert "xn--" in result["normalized_url"]
    assert any("internationalized" in indicator or "Unicode" in indicator for indicator in result["detected_indicators"])


@pytest.mark.parametrize(
    "url",
    ["https://exa mple.com/", "https://example.com/\npath", "https://example.com/\x01", "https://example.com/\x7f"],
)
def test_whitespace_or_control_characters_make_input_invalid(url):
    result = analyze_url(url)

    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None


def test_common_multilabel_suffix_does_not_count_registrable_domain_as_subdomain():
    result = analyze_url("https://example.co.uk/login")

    assert result["analysis_status"] == "analyzed"
    assert not any("subdomains" in indicator for indicator in result["detected_indicators"])


def test_subdomains_before_common_multilabel_suffix_are_counted():
    result = analyze_url("https://a.b.c.d.example.co.uk/login")

    assert any("4 subdomains" in indicator for indicator in result["detected_indicators"])
