#!/bin/bash

source ~/.profile

echo "Clearing GoShimmer Folders"
DIR="/tmp/peerdb"
if [ -d "$DIR" ]; then
  rm -rf $DIR
fi

DIR="/tmp/mainnetdb"
if [ -d "$DIR" ]; then
  rm -rf $DIR
fi

echo "Pulling latest GoShimmer repository"
cd /opt/goshimmer
git reset --hard
git clean -fdx
git pull

echo "Downloading the latest snapshot"
wget -q -O snapshot.bin https://dbfiles-goshimmer.s3.eu-central-1.amazonaws.com/snapshots/nectar/snapshot-latest.bin

echo "Changing interface to 0.0.0.0 to make services accessible"
cp /config.default.json /opt/goshimmer/config.default.json
mv config.default.json config.json
sed -i 's/127.0.0.1/0.0.0.0/' config.json

echo "Copying plugins"
cp -R /proxdag /usr/local/go/src/

echo "Add plugin to research.go"
cp /research_sample.go /opt/goshimmer/plugins/research.go

echo "Building GoShimmer"
./scripts/build.sh

echo "Running GoShimmer"
./goshimmer --skip-config=true \
            --autoPeering.entryNodes=2PV5487xMw5rasGBXXWeqSi4hLz7r19YBt8Y1TGAsQbj@analysisentry-01.devnet.shimmer.iota.cafe:15626,5EDH4uY78EA6wrBkHHAVBWBMDt7EcksRq6pjzipoW15B@entry-0.devnet.tanglebay.com:14646,CAB87iQZR6BjBrCgEBupQJ4gpEBgvGKKv3uuGVRBKb4n@entry-1.devnet.tanglebay.com:14646 \
            --node.disablePlugins=portcheck \
            --node.enablePlugins=remotelog,networkdelay,spammer,prometheus,txstream,proxdag \
            --database.directory=/tmp/mainnetdb \
            --node.peerDBDirectory=/tmp/peerdb \
            --logger.level=info \
            --logger.disableEvents=false \
            --logger.remotelog.serverAddress=metrics-01.devnet.shimmer.iota.cafe:5213 \
            --drng.pollen.instanceID=1 \
            --drng.pollen.threshold=3 \
            --drng.pollen.committeeMembers=AheLpbhRs1XZsRF8t8VBwuyQh9mqPHXQvthV5rsHytDG,FZ28bSTidszUBn8TTCAT9X1nVMwFNnoYBmZ1xfafez2z,GT3UxryW4rA9RN9ojnMGmZgE2wP7psagQxgVdA4B9L1P,4pB5boPvvk2o5MbMySDhqsmC2CtUdXyotPPEpb7YQPD7,64wCsTZpmKjRVHtBKXiFojw7uw3GszumfvC4kHdWsHga \
            --drng.xTeam.instanceID=1339 \
            --drng.xTeam.threshold=4 \
            --drng.xTeam.committeeMembers=GUdTwLDb6t6vZ7X5XzEnjFNDEVPteU7tVQ9nzKLfPjdo,68vNzBFE9HpmWLb2x4599AUUQNuimuhwn3XahTZZYUHt,Dc9n3JxYecaX3gpxVnWb4jS3KVz1K1SgSK1KpV1dzqT1,75g6r4tqGZhrgpDYZyZxVje1Qo54ezFYkCw94ELTLhPs,CN1XLXLHT9hv7fy3qNhpgNMD6uoHFkHtaNNKyNVCKybf,7SmttyqrKMkLo5NPYaiFoHs8LE6s7oCoWCQaZhui8m16,CypSmrHpTe3WQmCw54KP91F5gTmrQEL7EmTX38YStFXx