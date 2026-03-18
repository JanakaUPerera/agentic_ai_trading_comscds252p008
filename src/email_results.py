from __future__ import annotations

import smtplib
from email.message import EmailMessage
from src.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    EMAIL_SENDER,
    EMAIL_RECIPIENT,
)


def validate_email_config() -> None:
    required = [
        SMTP_HOST,
        SMTP_PORT,
        SMTP_USERNAME,
        SMTP_PASSWORD,
        EMAIL_SENDER,
        EMAIL_RECIPIENT,
    ]
    if not all(required):
        raise ValueError("Missing SMTP or email configuration in .env")


def build_email_message(download_url: str, s3_uri: str) -> EmailMessage:
    """
    Build an email with an S3 download link.
    """
    message = EmailMessage()
    message["Subject"] = "Agentic AI Trading Workflow - Final Analysis Bundle"
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECIPIENT

    body = f"""Hello,

The final analysis bundle for the Agentic AI Trading Workflow project is ready.

S3 Location:
{s3_uri}

Temporary Download Link:
{download_url}

Please note:
- The download link is temporary and may expire.
- The S3 object remains stored in the configured bucket.

Best regards,
Agentic AI Trading Workflow
"""

    message.set_content(body)
    return message


def send_email_with_s3_link(download_url: str, s3_uri: str) -> None:
    """
    Send email containing S3 link for the final analysis bundle.
    """
    validate_email_config()
    message = build_email_message(download_url=download_url, s3_uri=s3_uri)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)

    print(f"S3 link email sent successfully to {EMAIL_RECIPIENT}")