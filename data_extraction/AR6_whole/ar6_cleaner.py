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

# remove . from et al.
entire_doc = re.sub(r'et al\.', 'et al', entire_doc)

# remove . from e.g.
entire_doc = re.sub(r'e\.g\.', 'eg', entire_doc)

# remove . from i.e.
entire_doc = re.sub(r'i\.e\.', 'ie', entire_doc)

# remove . from etc.
entire_doc = re.sub(r'etc\.', 'etc', entire_doc)

# remove . from Fig.
entire_doc = re.sub(r'Fig\.', 'Fig', entire_doc)

# remove . from Eq.
entire_doc = re.sub(r'Eq\.', 'Eq', entire_doc)

# remove . from Sect.
entire_doc = re.sub(r'Sect\.', 'Sect', entire_doc)

# wherever there is a full stop followed by a capital letter, add a space after the full stop
entire_doc = re.sub(r'\.([A-Z])', r'. \1', entire_doc)

# write to file
with open("data_extraction/AR6_whole/AR6_whole_cleaned.txt", "w") as f:
    f.write(entire_doc)








