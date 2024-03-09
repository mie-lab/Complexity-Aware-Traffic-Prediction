# Create the combined.csv with just the header from the first file
head -n 1 london-4-2-55-validation_csv_hard_1.csv > combined.csv

# Maximum number of hard and easy files
max_num=5

# Loop to alternate between "hard" and "easy" and append the content without headers
for i in $(seq 1 $max_num); do
    if [ -e "london-4-2-55-validation_csv_hard_$i.csv" ]; then
        tail -n +2 "london-4-2-55-validation_csv_hard_$i.csv" >> combined_temp.csv
    fi

    if [ -e "london-4-1-55-validation_csv_easy_$i.csv" ]; then
        tail -n +2 "london-4-1-55-validation_csv_easy_$i.csv" >> combined_temp.csv
    fi
done

# Fixing the epoch column
awk -F, 'BEGIN{OFS=","} {print NR-1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17}' combined_temp.csv >> combined.csv

# Cleanup
rm combined_temp.csv

