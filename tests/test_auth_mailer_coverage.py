from urllib.parse import parse_qs, urlparse

import pytest

from doubtdesk.auth import mailer


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.credentials = None
        self.messages = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, user, password):
        self.credentials = (user, password)

    def send_message(self, message):
        self.messages.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_USER", "mailer@example.test")
    monkeypatch.setenv("SMTP_PASSWORD", "not-a-real-password")
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    monkeypatch.delenv("AUTH_FRONTEND_URL", raising=False)
    return FakeSMTP.instances


def test_verification_email_sends_correct_link_and_uses_secure_smtp(smtp):
    token = "plus+/= token&?"
    mailer.send_verification_email("student@example.test", token)
    assert len(smtp) == 1
    sender = smtp[0]
    assert (sender.host, sender.port, sender.timeout) == ("smtp.example.test", 465, 15)
    assert sender.credentials == ("mailer@example.test", "not-a-real-password")
    message = sender.messages[0]
    assert message["From"] == "mailer@example.test"
    assert message["To"] == "student@example.test"
    assert message["Subject"] == "Verify your DoubtDesk account"
    text = message.get_content()
    url = next(word for word in text.split() if word.startswith("http://"))
    parsed = urlparse(url)
    assert parsed.path == "/verify-email"
    assert parse_qs(parsed.query)["token"] == [token]
    assert "10 minutes" in text
    assert "not-a-real-password" not in text


def test_password_reset_email_custom_frontend_sender_and_port(smtp, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "2465")
    monkeypatch.setenv("SMTP_FROM", "noreply@example.test")
    monkeypatch.setenv("AUTH_FRONTEND_URL", "https://app.example.test/")
    mailer.send_password_reset_email("student@example.test", "reset-a-token")
    sender = smtp[0]
    assert sender.port == 2465
    message = sender.messages[0]
    assert message["From"] == "noreply@example.test"
    assert message["To"] == "student@example.test"
    assert message["Subject"] == "Reset your DoubtDesk password"
    assert "https://app.example.test/reset-password?token=reset-a-token" in message.get_content()
    assert "10 minutes" in message.get_content()


def test_email_credentials_are_required(smtp, monkeypatch):
    monkeypatch.delenv("SMTP_HOST")
    with pytest.raises(KeyError, match="SMTP_HOST"):
        mailer.send_verification_email("student@example.test", "token")
