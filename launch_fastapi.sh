ray stop --force | true
sleep 2
ray start --head --port 8080 --include-dashboard false
sleep 2
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
