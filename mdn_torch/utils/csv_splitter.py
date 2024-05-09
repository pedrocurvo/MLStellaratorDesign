import csv
import os

def csv_splitter(input_file, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the input CSV file
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the header row

        # Split the data into 20 equal parts
        chunk_size = sum(1 for _ in reader) // 13
        csvfile.seek(0)  # Reset file pointer to beginning

        # Write the data to multiple output files
        for i in range(13):
            output_file = os.path.join(output_dir, f'output_{i+1}.csv')
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(headers)  # Write header row
                for _ in range(chunk_size):
                    try:
                        row = next(reader)
                        writer.writerow(row)
                    except StopIteration:
                        break
