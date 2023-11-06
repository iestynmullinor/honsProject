# cleans up the AR6_whole.txt file by making the following changes:
# removes all references 
# removes everything contained in curly brackets
# adds a space after every full stop

import re

with open("data_extraction/AR6_whole/AR6_whole.txt", "r") as f:
    entire_doc = f.read()

# remove all references
entire_doc = re.sub(r'\[.*?\]', '', entire_doc)

# remove everything contained in curly brackets
entire_doc = re.sub(r'\{.*?\}', '', entire_doc)

# remove all references of format ("name", year) where year is 4 digits
entire_doc = re.sub(r'\(.*?\d{4}.*?\)', '', entire_doc)

# remove all references of format ("name et al." year) where year is 4 digits
entire_doc = re.sub(r'\(.*?et al\..*?\d{4}.*?\)', '', entire_doc)

# write to file
with open("data_extraction/AR6_whole/AR6_whole_cleaned.txt", "w") as f:
    f.write(entire_doc)








