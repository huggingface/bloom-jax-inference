ray stop --force | true
sleep 2
ray start --head --port 8080 --include-dashboard false
python run.py
ray stop --force | true
sleep 2