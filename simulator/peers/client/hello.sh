#!/bin/bash

echo "Welcome from $MY_NAME"

echo "$MY_NAME Clearing up environment"
rm -rf /ldb
rm -rf /temp/*

echo "$MY_NAME Making temporary directory"
mkdir /temp

echo "$MY_NAME Running listener"
python3 listener.py &

echo "$MY_NAME Sleeping for 6 seconds"
sleep 6

echo "$MY_NAME Learning"
# while :
# do
  python3 main.py -d MNIST -r 2
#  echo "$MY_NAME Sleeping for 5 seconds"
#   sleep 5
# done


# To keep the container running for testing purposes
tail -f /dev/null
