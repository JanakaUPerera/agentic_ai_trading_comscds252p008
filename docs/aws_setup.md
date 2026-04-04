# AWS Cloud Setup

## Overview
This project uses AWS cloud services to support a complete cloud-based trading analytics workflow.

## Services Used

### Amazon EC2
Amazon EC2 is used as the computation environment. The full Python workflow runs on an Ubuntu EC2 instance.

### Amazon S3
Amazon S3 stores raw data, processed data, generated outputs, figures, and tables.

### Amazon RDS PostgreSQL
Amazon RDS PostgreSQL stores structured analytical summaries such as:
- news sentiment summary
- decision summary
- risk summary
- backtest summary

## Execution Flow
1. The project repository is cloned to EC2.
2. Python environment and dependencies are installed on EC2.
3. The full workflow is run using `python main.py`.
4. Generated files are uploaded to S3.
5. Structured summary tables are uploaded to RDS.

## Environment Variables

Create a `.env` file in the project root based on `.env.example` and fill in your own credentials.

> Note: The repository does not include real secrets, API keys, database passwords, or private cloud credentials for security reasons.

## EC2 Setup Commands

```bash
sudo apt update
sudo apt install python3-pip python3-venv git -y

git clone https://github.com/JanakaUPerera/agentic_ai_trading_comscds252p008.git
cd agentic_ai_trading_comscds252p008

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py