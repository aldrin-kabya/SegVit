#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval mIoU  --launcher pytorch ${@:3}