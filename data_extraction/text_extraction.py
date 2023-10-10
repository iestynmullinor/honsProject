import urllib.request
import os
from bs4 import BeautifulSoup
import shutil


URL = "https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-17/"
DIRECTORY = "/home/iestyn/honsProject/data_extraction/KB/wg3/chapter17/"

# classes to search for
class_names = [
    'h1-siblings', 'h2-siblings', 'h3-siblings', 'h4-siblings', 'h5-siblings', 'h6-siblings',
    'h1-container', 'h2-container', 'h3-container', 'h4-container', 'h5-container', 'h6-container'
]

sibling_names = ['h1-siblings', 'h2-siblings', 'h3-siblings', 'h4-siblings', 'h5-siblings', 'h6-siblings']
container_names = ['h1-container', 'h2-container', 'h3-container', 'h4-container', 'h5-container', 'h6-container']

bad_sections = ["References", "Acknowledgements", "Frequently Asked Questions", "Frequently Asked Questions"]

def create_dir_rec(section, parent_dir, container_level):


    # if text section, it is written to text file
    if section['class'][0] in sibling_names and len(section['class'])==1 :
        f = open(parent_dir + "/" + "section_text.txt", "w+")
        f.write(section.get_text())
        f.close()
    
    # if subsection, create new directory 
    elif section['class'][0] == container_names[container_level] and len(section['class'])==1 :
        title = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        if title:

            sub_section_name = title.get_text()

            # certain sections don't format well or are uselss
            if not any([bad_section in sub_section_name for bad_section in bad_sections]):

                # remove ones which include "Expand Section"
                sub_section_name = sub_section_name.replace("Expand section","")
                sub_section_name = sub_section_name.replace("/"," ")

                # create a directory for the subsection
                new_dir = os.path.join(parent_dir, sub_section_name)


                os.mkdir(new_dir)


                for sub_section in section.find_all(class_=class_names):
                    create_dir_rec(sub_section, new_dir, container_level+1)




        

def extract_chapter(url, home_dir):

    # needed since website blocks without it
    req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0'}
)
    webpage = urllib.request.urlopen(req)
    html = webpage.read()
    soup = BeautifulSoup(html, features="html.parser")

    # find the main section of text
    main_body = soup.find('div', class_='col-lg-10 col-12 offset-lg-0')

    for sub_section in main_body.find_all('div', class_='h1-container'):
        create_dir_rec(sub_section, home_dir, 0)

# DELETE DIRECTORY IF ALREADY EXISTS
def clean_dirs(directory_path):

    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    os.mkdir(directory_path)

if __name__=="__main__":
    clean_dirs(DIRECTORY)
    extract_chapter(URL, DIRECTORY)