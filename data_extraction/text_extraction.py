import urllib.request
import os
from bs4 import BeautifulSoup
import shutil

# THIS WEB SCRAPER WORKS FOR WG1, WG2, WG3, AND SYNTHESIS LONGER REPORT
# DOES NOT WORK FOR SUMMARY OF HEADLINES OR SUMMARY FOR POLICYMAKERS

ALL_URLS = ["https://www.ipcc.ch/report/ar6/syr/longer-report/", 
            
"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-1/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-2/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-3/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-4/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-5/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-6/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-8/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-10/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-11/",

"https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-12/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-1/ ",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-2/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-3/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-4/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-5/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-6/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-7/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-8/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-9/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-10/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-11/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-12/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-13/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-14/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-15/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-16/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-17/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-18/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp1/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp2/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp3/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp4/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp5/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp6/",

"https://www.ipcc.ch/report/ar6/wg2/chapter/ccp7/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-1/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-2/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-3/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-4/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-5/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-6/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-7/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-8/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-9/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-10/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-11/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-12/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-13/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-14/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-15/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-16/",

"https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-17/"]




ALL_DIRECTORIES = ["/synthesis","/wg1/chapter1","/wg1/chapter2","/wg1/chapter3","/wg1/chapter4","/wg1/chapter5",
                   "/wg1/chapter6","/wg1/chapter7","/wg1/chapter8","/wg1/chapter9","/wg1/chapter10", "/wg1/chapter11",
                   "/wg1/chapter12","/wg2/chapter1", "/wg2/chapter2", "/wg2/chapter3", "/wg2/chapter4", "/wg2/chapter5",
                   "/wg2/chapter6","/wg2/chapter7","/wg2/chapter8","/wg2/chapter9","/wg2/chapter10","/wg2/chapter11",
                   "/wg2/chapter12", "/wg2/chapter13","/wg2/chapter14","/wg2/chapter15","/wg2/chapter16","/wg2/chapter17",
                   "/wg2/chapter18","/wg2/ccp1", "/wg2/ccp2", "/wg2/ccp3", "/wg2/ccp4", "/wg2/ccp5", "/wg2/ccp6","/wg2/ccp7",
                   "/wg3/chapter1", "/wg3/chapter2", "/wg3/chapter3", "/wg3/chapter4", "/wg3/chapter5", "/wg3/chapter6", "/wg3/chapter7", 
                   "/wg3/chapter8","/wg3/chapter9", "/wg3/chapter10", "/wg3/chapter11", "/wg3/chapter12","/wg3/chapter13",
                   "/wg3/chapter14", "/wg3/chapter15", "/wg3/chapter16","/wg3/chapter17"]


BASE_DIRECTORY = "/home/iestyn/honsProject/data_extraction/KB"

#use these to just extract one page into one directory
URL = "https://www.ipcc.ch/report/ar6/syr/longer-report/"
DIRECTORY = "/home/iestyn/honsProject/data_extraction/KB/synthesis"

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

# reads one chapter into directory
def populate_chapter(url, directory):
    clean_dirs(directory)
    extract_chapter(url, directory)

if __name__=="__main__":
    
    #make directories if not already exist
    if not os.path.exists(BASE_DIRECTORY + "/wg1"):
        os.mkdir(BASE_DIRECTORY + "/wg1")
    if not os.path.exists(BASE_DIRECTORY + "/wg2"):
        os.mkdir(BASE_DIRECTORY + "/wg2")
    if not os.path.exists(BASE_DIRECTORY + "/wg3"):
        os.mkdir(BASE_DIRECTORY + "/wg3")

    # READ IN EVERY PAGE    
    for i in range(len(ALL_URLS)):
        populate_chapter(ALL_URLS[i], BASE_DIRECTORY + ALL_DIRECTORIES[i])