import csv

# Input and output file names
directory="/home/studium/Documents/Code/rdcn-throughput/Goran_Bachelor_Code/"

input_file =  directory + 'output2NoFloor.csv'
output_file = directory + 'output2NoRounding.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=' ')
    writer = csv.writer(outfile, delimiter=' ')
    
    # Get the header row
    header = next(reader)
    writer.writerow(header)  # Write the header to the output file
    
    # Process each row
    for row in reader:
        if row[3] != "Rounding":  # Check if the 'Alg' column (index 3) is not "Floor"
            writer.writerow(row)

print(f"Rows with 'Floor' in the Alg column have been removed. Modified file saved as: {output_file}")