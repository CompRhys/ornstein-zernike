#!/usr/bin/env bash
echo "Run Processing Script on Files"

number="`wc -l < inputs.txt`"
echo $number
for i in $(seq 1 $number); do
	input="`awk "NR==$i" inputs.txt`"
	echo "$input"
	# echo "python process.py $input"
done 