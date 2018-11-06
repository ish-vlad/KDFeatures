#!/usr/bin/env bash

cd ../../store
mkdir buf/ buf/results buf/datasets

mv ../datasets/MNIST_2D/Rotation buf/datasets/
mv ../datasets/MNIST_2D/Translation buf/datasets/
mv ../datasets/MNIST_2D/Rotation_and_Translation buf/datasets/

cp -r ../results/MNIST_2D/* buf/results/

zip -r v1.3.zip buf
rm -r buf

cd ../workspace/scripts