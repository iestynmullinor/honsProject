import os

# still need to make a better version of this script because it is shit

# Function to recursively search for .txt files and append their contents
def append_txt_files(directory, output_file):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    output_file.write(file.read())

# Specify the directory you want to search in and the output file name
input_directory = 'data_extraction/KB'
output_file_name = 'data_extraction/AR6_whole/ipcc_ar6.txt'

# Open the output file for writing
with open(output_file_name, 'w') as output_file:
    append_txt_files(input_directory, output_file)
    

print(f'All .txt files from {input_directory} have been appended to {output_file_name}')
