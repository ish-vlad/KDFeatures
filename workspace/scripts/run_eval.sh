#!/usr/bin/env bash

cd ../
python2.7 make_eval_set.py
python2.7 eval_MNIST.py
cd scripts/