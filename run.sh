ray stop --force | true
sleep 2
ray start --head --port 8080
python run.py
ray stop --force | true
sleep 2