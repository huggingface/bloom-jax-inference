#!/usr/bin/env bash
ray stop --force | true
sleep 2
ray start --head --port 48975 --include-dashboard false
sleep 2
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000
