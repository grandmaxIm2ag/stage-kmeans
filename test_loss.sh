#! /bin/sh

while read line
do
    echo $line
    ./main.py $line
done < .main.txt

latexmk -pdf loss.tex
