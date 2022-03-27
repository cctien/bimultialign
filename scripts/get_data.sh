#!/usr/bin/env sh

REPO=$PWD
SCRIPTS=$REPO/scripts
DATA_DIR=$REPO/data
mkdir -p $DATA_DIR

# Get BUCC2018 and Tatoeba data with scripts from [Xtreme by Hu et al.](https://github.com/google-research/xtreme)
bash $SCRIPTS/xtreme/download_data.sh
