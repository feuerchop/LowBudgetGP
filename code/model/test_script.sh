#!/bin/bash

for i in 0 0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10
do
    python run_test.py real mlp 10 250 $i 10 >> ./results_new/out_mlp_$i.log &
done
