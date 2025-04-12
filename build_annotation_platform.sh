#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 TAG"
    echo "TAG: Docker image tag to use"
    exit 1
fi

TAG=$1

GIT_COMMIT="$(git rev-parse HEAD)"
TIMESTAMP="$(date)"

docker build --build-arg GIT_COMMIT="$GIT_COMMIT" --build-arg TIMESTAMP="$TIMESTAMP" -t abdeshpande/semantic-grasping-annotation:$TAG .
docker push abdeshpande/semantic-grasping-annotation:$TAG
