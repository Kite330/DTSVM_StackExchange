#!/bin/bash

DATA_DIR=$HOME/graduate/analysis
SRC_DIR=/home/paruru/graduate/doc2vec.py

PSTATS_DATA=$DATA_DIR/d2v.pstats
PNG_DATA=$DATA_DIR/d2v.png

time valgrind --tool=massif python -m cProfile -o $PSTATS_DATA $SRC_DIR
python -m gprof2dot -f pstats $PSTATS_DATA | dot -Tpng -o $PNG_DATA
