import os
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode

def _send(to_email: str, subject: str, body: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = to_email
    message.set_content(body)

    with smtplib.SMTP_SSL(
        smtp_host,
        smtp_port,
        timeout=15,
    ) as smtp:
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(message)


def send_verification_email(
    to_email: str,
    token: str,
) -> None:
    frontend_url = os.getenv(
        "AUTH_FRONTEND_URL",
        "http://localhost:5173",
    ).rstrip("/")

    verification_link = (
        f"{frontend_url}/verify-email?"
        f"{urlencode({'token': token})}"
    )

    _send(to_email, "Verify your DoubtDesk account",
        f"""
Welcome to DoubtDesk!

Please verify your email address by visiting:

{verification_link}

This link expires in 10 minutes.

If you didn't create this account, ignore this email.

DoubtDesk Team
""".strip()
    )



def send_password_reset_email(
    to_email: str,
    token: str,
) -> None:
    frontend_url = os.getenv(
        "AUTH_FRONTEND_URL",
        "http://localhost:5173",
    ).rstrip("/")

    reset_link = (
        f"{frontend_url}/reset-password?"
        f"{urlencode({'token': token})}"
    )

    _send(to_email, "Reset your DoubtDesk password",
        f"""
Hello!

We received a request to reset your DoubtDesk password.

Reset your password using this link:

{reset_link}

This link expires in 10 minutes.

If you didn't request this, ignore this email.

DoubtDesk Team
""".strip()
    )
