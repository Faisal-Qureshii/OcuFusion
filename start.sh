#!/usr/bin/env bash
set -e
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --reload
