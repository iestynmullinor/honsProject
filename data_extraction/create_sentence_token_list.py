import os
import doc_cleaner
from nltk import sent_tokenize
from nltk import word_tokenize
import pickle


TEXT = "section_text.txt"
INPUT_DIRECTORY = 'data_extraction/KB'
OUTPUT_FILE_NAME = 'sentence_similarity/data/sentence_section_pairs.txt'
PICKLE_FILE_NAME = 'sentence_similarity/data/sentence_section_pairs.pkl'
sentences_with_section = []
sentence_set = set()


def tokenize_section(section_text):
    cleaned_section = doc_cleaner.clean_section(section_text)
    sentences = sent_tokenize(cleaned_section)
    sentences = [s for s in sentences if len(word_tokenize(s)) >= 7 and s not in sentence_set] 
    
    # Add each sentence to the sentence_set
    for sentence in sentences:
        sentence_set.add(sentence)
    
    return sentences


def add_sections(directory):

    # check if section_test.txt file exists in directory
    if os.path.isfile(os.path.join(directory, TEXT)):
        with open(os.path.join(directory, TEXT), 'r') as f:
            contents = f.read()

            sentences = tokenize_section(contents)
            
            # Append (sentence, section) for each sentence in sentences
            for sentence in sentences:
                section = directory.split('KB')[1]
                sentences_with_section.append((sentence, section))

    for file in sorted(os.scandir(directory), key=lambda f: f.name):
        if os.path.isdir(file.path):
            add_sections(file.path)
    
    return sentences_with_section


if __name__=="__main__":
    
    sentence_section_pairs = add_sections(INPUT_DIRECTORY)
    #print(sentence_section_pairs[:10])

    # Write to pickle file
    with open(PICKLE_FILE_NAME, 'wb') as f:
        pickle.dump(sentence_section_pairs, f)
    
    # Write to txt file
    with open(OUTPUT_FILE_NAME, 'w') as f:
        for pair in sentence_section_pairs:
            f.write(f"SENTENCE: {pair[0]}\nSECTION: {pair[1]}\n \n")
