"""Basic smoke checks for the initial PocketShield package structure."""


def test_package_imports():
    import pocketshield
    import pocketshield.phishing

    assert pocketshield is not None
    assert pocketshield.phishing is not None
