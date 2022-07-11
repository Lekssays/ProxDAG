#!/bin/bash

# echo "python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc true -at  $1 -ap 50"
# python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc true -at  $1 -ap 50

# sleep 60

# echo "python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc true -at  $1 -ap 90"
# python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc true -at  $1 -ap 90

# sleep 60


# echo "python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc true -at  $1 -ap 90"
# python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc true -at  $1 -ap 90

# sleep 60

echo "python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc false -at  $1 -ap 50"
python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc false -at  $1 -ap 50

sleep 60

echo "python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc false -at  $1 -ap 90"
python3 simulator.py -d MNIST -p 100 -i 5 -al 100 -dc false -at  $1 -ap 90

sleep 60


echo "python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc false -at  $1 -ap 90"
python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc false -at  $1 -ap 90

sleep 60

echo "python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc false -at $1 -ap 50"
python3 simulator.py -d MNIST -p 100 -i 10 -al 0.05 -dc false -at $1 -ap 50
