#!/bin/bash

# Original file name
input_file="record_distribution_reverse.csv"

# Number of lines per file
lines_per_file=24

# Loop to split the file and rename the output files
for (( i=9; i>=5; i-- ))
do
    # Calculate the line range for sed
    # We need to adjust the start_line and end_line calculations to account for the reversed loop
    start_line=$(( (9-i) * lines_per_file + 1 ))
    end_line=$(( (10-i) * lines_per_file ))

    # Generate the new file name
    new_file="${input_file%.csv}_ph${i}.csv"

    # Use sed to extract the line range and create the new file
    sed -n "${start_line},${end_line}p" "$input_file" > "$new_file"
done

