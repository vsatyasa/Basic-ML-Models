#!/bin/sh

a=0
while [ "$a" -lt 10 ]    # this is loop1
do
   python3 NeuralNetwork.py 
   python3 NeuralNetwork2.py
   
   echo "\n"
   a=`expr $a + 1`
done
