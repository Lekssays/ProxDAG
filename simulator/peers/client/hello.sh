#!/bin/bash

echo "Welcome from $MY_NAME"

echo "$MY_NAME Clearing up environment"
rm -rf /ldb
rm -rf /temp/*

echo "$MY_NAME Making temporary directory"
mkdir /temp

echo "$MY_NAME Running listener"
python3 listener.py &

# To keep the container running for testing purposes
tail -f /dev/null
