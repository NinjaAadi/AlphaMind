#!/bin/bash
# Cron wrapper: run from rag_service directory (current dir = script location).
# Usage in crontab (3:16 AM IST): use full path to this script.
# Example: 16 3 * * * /Users/aaditya.pal/Documents/AlphaMind/rag_service/run_ingest_cron.sh >> /tmp/rag_ingest.log 2>&1
set -e
cd "$(dirname "$0")"
exec ./venv/bin/python -m run_ingest
