docker stop $(docker ps -a -q  --filter ancestor=lekssays/proxdag:latest)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/proxdag:latest)

redis-cli FLUSHALL