#!/usr/bin/env bash

# Adapted from [XTREME](https://github.com/google-research/xtreme/blob/master/scripts/download_data.sh).

REPO=$PWD
SCRIPTS=$REPO/scripts/xtreme/
DIR=$REPO/data/
mkdir -p $DIR

function download_bucc18() {
    base_dir=$DIR/bucc2018/
    cd $DIR
    for lg in zh ru de fr; do
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.training-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.training-gold.tar.bz2
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.sample-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.sample-gold.tar.bz2
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.test.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.test.tar.bz2
    done

    mv $base_dir/*/* $base_dir/
    for f in $base_dir/*test*; do mv $f ${f/test/train}; done
    for f in $base_dir/*sample*; do mv $f ${f/sample/dev}; done
    for f in $base_dir/*training*; do mv $f ${f/training/test}; done
    rm -rf $DIR/bucc2018*tar.bz2 $base_dir/{zh,ru,de,fr}-en/
    echo "Successfully downloaded data at $DIR/bucc2018" >>$DIR/download.log

    TL='en'
    for SL in fr ru zh de; do
        for sp in 'test' 'dev'; do
            for lg in "$SL" "$TL"; do
                FILE=$base_dir/${SL}-${TL}.${sp}.${lg}
                cut -f2 $FILE >$base_dir/${SL}-${TL}.${sp}.${lg}.txt
                cut -f1 $FILE >$base_dir/${SL}-${TL}.${sp}.${lg}.id
            done
        done
    done
}

function download_tatoeba() {
    base_dir=$DIR/tatoeba/
    wget https://github.com/facebookresearch/LASER/archive/refs/heads/main.zip
    unzip -qq -o main.zip -d $base_dir/
    mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
    rm -rf $base_dir/LASER-main/main.zip
    echo "Successfully downloaded data at $DIR/tatoeba" >>$DIR/download.log
}

download_bucc18
download_tatoeba
