#!/bin/bash

# Original file name
input_file="record_distribution.csv"

# Number of lines per file
lines_per_file=24

# Total number of lines in the file
total_lines=$(wc -l < "$input_file")

# Number of files to be created
num_files=$(( (total_lines + lines_per_file - 1) / lines_per_file ))

# Loop to split the file and rename the output files
for (( i=1; i<=num_files; i++ ))
do
    # Calculate the line range for sed
    start_line=$(( (i-1) * lines_per_file + 1 ))
    end_line=$(( i * lines_per_file ))

    # Generate the new file name
    new_file="${input_file%.csv}_ph${i}.csv"

    # Use sed to extract the line range and create the new file
    sed -n "${start_line},${end_line}p" "$input_file" > "$new_file"
done

