#!/bin/bash

head -n 1 ../raw/result_0.csv > result_combined.csv
for res in ../raw/result_*.csv;
do
   tail -n +2 $res >> result_combined.csv
done
