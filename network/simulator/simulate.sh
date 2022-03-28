#!/bin/bash

docker network create proxdag

cd ./goshimmer/
docker-compose up -d

sleep 60

cd ../peers/
docker-compose up -d
