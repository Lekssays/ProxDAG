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
            --node.enablePlugins=remotelog,networkdelay,spammer,prometheus,proxdag \
            --database.directory=/tmp/mainnetdb \
            --node.peerDBDirectory=/tmp/peerdb \
            --logger.level=info \
            --logger.disableEvents=false \
            --logger.remotelog.serverAddress=metrics-01.devnet.shimmer.iota.cafe:5213