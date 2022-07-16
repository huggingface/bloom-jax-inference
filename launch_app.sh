ray stop --force | true
sleep 2
ray start --head --port 8080 --include-dashboard false
sleep 2
python flask_app.py
