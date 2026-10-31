"""Unit tests for PocketShield's offline SMS scam heuristics."""

import pytest

from pocketshield.sms.sms_detector import analyze_sms


def test_normal_message_has_no_configured_scam_indicators():
    result = analyze_sms("Hi, I will meet you at the library at 5 pm.")

    assert result["analysis_status"] == "analyzed"
    assert result["risk_score"] == 0
    assert result["risk_level"] == "Low"
    assert result["is_suspicious"] is False
    assert result["reasons"] == ["No configured scam indicators were found; this does not prove the message is safe."]


def test_prize_claim_is_detected():
    result = analyze_sms("Congratulations! You have won a lottery prize. Claim your reward now.")

    assert result["risk_score"] == 25
    assert result["risk_level"] == "Low"
    assert any("prize" in reason for reason in result["reasons"])


def test_fake_bank_alert_with_verification_and_urgency_is_suspicious():
    result = analyze_sms("Urgent: Your bank account is locked. Verify your account immediately or it will be suspended.")

    assert result["risk_score"] == 35
    assert result["risk_level"] == "Medium"
    assert result["is_suspicious"] is True
    assert any("account verification" in reason for reason in result["reasons"])
    assert any("urgency" in reason for reason in result["reasons"])


def test_request_for_otp_is_detected_but_mention_alone_is_not():
    request = analyze_sms("Reply with your OTP to verify your account now.")
    mention = analyze_sms("Your bank OTP was used to sign in; this is just an alert.")

    assert any("OTP" in reason for reason in request["reasons"])
    assert request["risk_score"] >= 35
    assert mention["risk_score"] == 0
    assert mention["is_suspicious"] is False


def test_shortened_link_is_detected_without_visiting_it():
    result = analyze_sms("Track your parcel here: https://bit.ly/parcel123")

    assert result["risk_score"] == 30
    assert result["is_suspicious"] is True
    assert any("link" in reason for reason in result["reasons"])


def test_urgent_payment_request_accumulates_indicators():
    result = analyze_sms("Pay a $500 fee immediately or your account will be closed.")

    assert result["risk_score"] == 40
    assert result["risk_level"] == "Medium"
    assert result["is_suspicious"] is True
    assert any("send or pay money" in reason for reason in result["reasons"])
    assert any("deadline" in reason for reason in result["reasons"])


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_or_whitespace_message_is_invalid(message):
    result = analyze_sms(message)

    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None
    assert result["risk_level"] == "Unknown"
    assert result["is_suspicious"] is None
    assert result["reasons"]


@pytest.mark.parametrize("message", [None, 123, ["OTP", "please send"]])
def test_non_string_message_is_invalid(message):
    result = analyze_sms(message)

    assert result["analysis_status"] == "invalid"
    assert result["risk_score"] is None
    assert result["risk_level"] == "Unknown"
    assert result["is_suspicious"] is None
    assert result["reasons"]


def test_safety_warning_about_otp_is_not_treated_as_a_credential_request():
    result = analyze_sms("Never share your OTP or PIN with anyone over SMS.")

    assert result["risk_score"] == 0
    assert result["is_suspicious"] is False


def test_actionable_prize_redemption_is_detected_but_ordinary_prize_talk_is_not():
    offer = analyze_sms("You were selected to receive a £900 prize. Call 09061234567 now to claim it.")
    ordinary = analyze_sms("I hope you win the office raffle this year!")

    assert offer["risk_score"] >= 30
    assert offer["is_suspicious"] is True
    assert any("redeem" in reason for reason in offer["reasons"])
    assert ordinary["risk_score"] < 30
    assert ordinary["is_suspicious"] is False


def test_premium_rate_text_offer_needs_both_rate_and_response_action():
    offer = analyze_sms("Text PLAY to 85222 now. This service costs 150p per message.")
    ordinary = analyze_sms("Please text me when you arrive; the ticket cost 150p.")

    assert offer["risk_score"] >= 30
    assert offer["is_suspicious"] is True
    assert any("paid text" in reason for reason in offer["reasons"])
    assert ordinary["risk_score"] < 30
    assert ordinary["is_suspicious"] is False


def test_recurring_ringtone_charge_needs_service_charge_and_enrollment_context():
    offer = analyze_sms("Your ringtone subscription will cost £5/month. Reply YES to activate.")
    ordinary = analyze_sms("My music subscription costs £5 per month.")

    assert offer["risk_score"] >= 30
    assert offer["is_suspicious"] is True
    assert ordinary["risk_score"] < 30
    assert ordinary["is_suspicious"] is False


def test_targeted_promotional_offer_needs_benefit_item_and_redemption_action():
    offer = analyze_sms("You are specially selected for a free camera phone. Call now to claim your upgrade.")
    ordinary = analyze_sms("This phone plan includes free calls on weekends.")

    assert offer["risk_score"] >= 30
    assert offer["is_suspicious"] is True
    assert any("unsolicited offer" in reason for reason in offer["reasons"])
    assert ordinary["risk_score"] < 30
    assert ordinary["is_suspicious"] is False


def test_premium_rate_callback_lure_requires_context_and_premium_number():
    lure = analyze_sms("Important customer service announcement: call 09061234567 now.")
    ordinary = analyze_sms("Courier arriving today; call our local office if needed on 0123456789.")

    assert lure["risk_score"] >= 30
    assert lure["is_suspicious"] is True
    assert any("premium-rate phone number" in reason for reason in lure["reasons"])
    assert ordinary["risk_score"] < 30
    assert ordinary["is_suspicious"] is False


def test_premium_rate_scam_warning_does_not_trigger_new_paid_service_signals():
    result = analyze_sms(
        "Scam warning: never respond to WIN texts from 85222; this premium-rate scam charges 150p per message."
    )

    assert result["risk_score"] < 30
    assert result["is_suspicious"] is False


def test_related_paid_service_clues_count_as_one_signal_category():
    result = analyze_sms(
        "Subscribe to our ringtone service: text TONE to 85222; 150p per message and £5 per week."
    )

    paid_service_reasons = [reason for reason in result["reasons"] if "paid text or subscription" in reason]
    assert len(paid_service_reasons) == 1
    assert result["risk_score"] == 30


def test_simple_prize_claim_keeps_legacy_subthreshold_score():
    result = analyze_sms("Congratulations! You have won a lottery prize. Claim your reward now.")

    # A claim phrase alone has no separate phone, text, or code redemption action.
    assert result["risk_score"] == 25
    assert result["is_suspicious"] is False
