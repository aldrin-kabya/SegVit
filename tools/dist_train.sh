#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG --work-dir $WORK_DIR ${@:3}