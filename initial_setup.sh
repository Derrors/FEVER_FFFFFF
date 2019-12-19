#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd . > /dev/null

cd ${CURRENT_DIR}/../jack
python -m pip install -e .[tf]
bash ./data/GloVe/download.sh

popd > /dev/null

pushd . > /dev/null

cd ${CURRENT_DIR}/../fever-baselines
pip install -r requirements.txt
popd > /dev/null

