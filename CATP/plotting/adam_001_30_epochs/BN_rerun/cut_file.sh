#!/bin/bash

# File to be processed
input_file="record_distribution_application_experiments_BN.csv"

# Filter the file
grep "MP" "$input_file" > filtered_file.csv

# Get the total number of lines in the filtered file
total_lines=$(wc -l < filtered_file.csv)

# Calculate the number of lines per file (assuming equal split)
lines_per_file=$((total_lines / 8))
[ $((total_lines % 8)) -ne 0 ] && ((lines_per_file++))

# Split the file and rename the parts
split -l $lines_per_file filtered_file.csv temp_part_

# Naming the split files
mv temp_part_aa BN_1_log.csv
mv temp_part_ab No_BN_1_log.csv
mv temp_part_ac BN_2_log.csv
mv temp_part_ad No_BN_2_log.csv
mv temp_part_ae BN_3_log.csv
mv temp_part_af No_BN_3_log.csv
mv temp_part_ag BN_4_log.csv
mv temp_part_ah No_BN_4_log.csv

# Remove the temporary filtered file
rm filtered_file.csv

