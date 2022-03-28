#!/bin/bash

docker create network proxdag

cd ./goshimmer/
docker-compose up

cd ../peers/
docker-compose up
