from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    BASE_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    S3_BUCKET_NAME,
)


def create_s3_client():
    """
    Create and return an S3 client.
    Uses IAM role automatically on EC2 if access keys are not provided.
    """
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "s3",
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    return boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def collect_files_for_upload() -> list[Path]:
    """
    Collect relevant project output files to upload to S3.
    """
    candidate_dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR / "figures",
        REPORTS_DIR / "tables",
    ]

    files: list[Path] = []

    for directory in candidate_dirs:
        if directory.exists():
            files.extend([path for path in directory.rglob("*") if path.is_file()])

    return files


def build_s3_key(file_path: Path) -> str:
    """
    Build the S3 object key relative to the project base directory.
    """
    relative_path = file_path.relative_to(BASE_DIR)
    return str(relative_path).replace("\\", "/")


def upload_file_to_s3(file_path: Path, bucket_name: str, s3_key: str) -> str:
    """
    Upload a single file to S3 and return the S3 URI.
    """
    client = create_s3_client()

    try:
        client.upload_file(str(file_path), bucket_name, s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"Uploaded {file_path} to {s3_uri}")
        return s3_uri
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"Failed to upload {file_path}: {error}") from error


def upload_project_outputs_to_s3() -> None:
    """
    Upload collected project files to S3.
    """
    files = collect_files_for_upload()

    if not files:
        print("No files found for upload.")
        return

    print(f"Uploading {len(files)} files to S3 bucket: {S3_BUCKET_NAME}")

    for file_path in files:
        s3_key = build_s3_key(file_path)
        upload_file_to_s3(file_path, S3_BUCKET_NAME, s3_key)

    print("S3 upload completed.")


def generate_presigned_download_url(
    s3_key: str,
    expiration_seconds: int = 86400,
) -> str:
    """
    Generate a temporary presigned download URL for an S3 object.
    Default expiry: 24 hours.
    """
    client = create_s3_client()

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET_NAME,
                "Key": s3_key,
            },
            ExpiresIn=expiration_seconds,
        )
        return url
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"Failed to generate presigned URL for {s3_key}: {error}") from error


def upload_bundle_and_get_link(
    bundle_file: Path = OUTPUTS_DIR / "final_analysis_bundle.zip",
    expiration_seconds: int = 86400,
) -> tuple[str, str]:
    """
    Upload the final ZIP bundle and return both S3 URI and presigned URL.
    """
    if not bundle_file.exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_file}")

    s3_key = build_s3_key(bundle_file)
    s3_uri = upload_file_to_s3(bundle_file, S3_BUCKET_NAME, s3_key)
    presigned_url = generate_presigned_download_url(
        s3_key=s3_key,
        expiration_seconds=expiration_seconds,
    )

    print(f"Generated presigned URL for bundle: {s3_key}")
    return s3_uri, presigned_url


if __name__ == "__main__":
    try:
        upload_project_outputs_to_s3()
    except Exception as e:
        print(f"Error during S3 upload: {e}")