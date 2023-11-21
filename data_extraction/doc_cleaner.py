# cleans up the AR6_whole.txt file by making the following changes:
# removes all references 
# removes everything contained in curly brackets
# adds a space after every full stop

import re

def clean_section(section):

    # remove all references
    section = re.sub(r'\[.*?\]', '', section)

    # remove everything contained in curly brackets
    section = re.sub(r'\{.*?\}', '', section)

    # remove all references of format ("name", year) where year is 4 digits
    section = re.sub(r'\(.*?\d{4}.*?\)', '', section)

    # remove all references of format ("name et al." year) where year is 4 digits
    section = re.sub(r'\(.*?et al\..*?\d{4}.*?\)', '', section)

    # remove . from et al.
    section = re.sub(r'et al\.', 'et al', section)

    # remove . from e.g.
    section = re.sub(r'e\.g\.', 'eg', section)

    # remove . from i.e.
    section = re.sub(r'i\.e\.', 'ie', section)

    # remove . from etc.
    section = re.sub(r'etc\.', 'etc', section)

    # remove . from Fig.
    section = re.sub(r'Fig\.', 'Fig', section)

    # remove . from Eq.
    section = re.sub(r'Eq\.', 'Eq', section)

    # remove . from Sect.
    section = re.sub(r'Sect\.', 'Sect', section)

    # wherever there is a full stop followed by a capital letter, add a space after the full stop
    section = re.sub(r'\.([A-Z])', r'. \1', section)

    # remove full stops that do not have a character immediately before them
    section = re.sub(r'(?<!\w)\.', '', section)
    
    # reduce multiple consecutive spaces to a single space
    section = re.sub(r' +', ' ', section)

    

    return section

    








