import os

TEXT = "section_text.txt"
INPUT_DIRECTORY = 'data_extraction/KB'
OUTPUT_FILE_NAME = 'data_extraction/AR6_whole/AR6_whole.txt'


def add_sections(directory, output_file):
    
    # check if "section_text.txt" exists in the directory, then append the contents of it to output_file
    
    if os.path.isfile(os.path.join(directory, TEXT)):
        with open(os.path.join(directory, TEXT), 'r') as f:
            contents = f.read()

        with open(output_file, 'a') as f:
            f.write(" ")
            f.write(contents)

    for file in sorted(os.scandir(directory), key=lambda f: f.name):
        if os.path.isdir(file.path):
            add_sections(file.path, output_file)
        

if __name__=="__main__":
    if os.path.isfile(OUTPUT_FILE_NAME):
        os.remove(OUTPUT_FILE_NAME)
    else:
        print(f"File {OUTPUT_FILE_NAME} not found")
    add_sections(INPUT_DIRECTORY, OUTPUT_FILE_NAME)

