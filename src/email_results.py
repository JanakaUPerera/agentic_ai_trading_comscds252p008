from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_with_s3_link(
    download_url: str,
    s3_uri: str,
    recipient_override: str | None = None,
) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_sender = os.getenv("EMAIL_SENDER") or smtp_username
    default_recipient = os.getenv("EMAIL_RECIPIENT")

    recipient = recipient_override or default_recipient
    if not recipient:
        raise ValueError("No recipient email address provided.")

    if not smtp_host or not smtp_username or not smtp_password or not email_sender:
        raise ValueError("Missing SMTP configuration in environment variables.")

    subject = "Agentic AI Trading Workflow Results"
    body = f"""
Hello,

The full Agentic AI Trading Workflow has completed successfully.

S3 URI:
{s3_uri}

Download Link:
{download_url}

Regards,
Agentic AI Trading Workflow
""".strip()

    message = MIMEMultipart()
    message["From"] = email_sender
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(email_sender, recipient, message.as_string())

    print(f"Email sent successfully to {recipient}")