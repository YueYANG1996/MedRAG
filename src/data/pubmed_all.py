import os
import re
import json
from tqdm import tqdm
import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def split_text_into_chunks(text, max_chunk_length=1000): 
    # Split the text into sentences using regex that captures sentence end characters
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the max_chunk_length
        if len(current_chunk) + len(sentence) > max_chunk_length:
            # If adding the sentence exceeds the max length, finish the current chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # If not, add the sentence to the chunk
            current_chunk += " " + sentence

    # Add the last chunk in case there's any remaining sentences not added
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # if the last chunk is too short, merge it with the previous chunk
    if len(chunks) > 1 and len(chunks[-1]) < 300:
        chunks[-2] = concat(chunks[-2], chunks[-1])
        chunks.pop()

    return chunks


def process_files(files, chunk_id):
    save_text = []
    snippet_id = 0
    print(f"Processing chunk {chunk_id}")
    for file in files:
        file_id = file.split(".")[0]
        test_file = json.load(open(f"/nlp/data/PMC/OA/jsons/{file}", "r"))
        useful_sections = []
        for key in test_file.keys():
            if key == 'Supplementary Material': break
            useful_sections.append(key)
        
        section2content = {}
        for section in useful_sections:
            if type(test_file[section]) == str:
                section2content[section] = test_file[section]
            elif type(test_file[section]) == dict:
                for subsection, content in test_file[section].items():
                    if type(content) == str: section2content[subsection] = content
            else:
                continue

        for section, content in section2content.items():
            content = content.replace("\n", " ")
            if len(content) > 0:
                texts = split_text_into_chunks(content)
                for text in texts:
                    if len(text) < 100: continue
                    save_text.append(json.dumps({"id": f"{chunk_id}_{snippet_id}", "title": section, "contents": text, "PMC_ID": file_id}))
                    snippet_id += 1

    with open(f"../../corpus/pubmed_all/chunk/{chunk_id}.jsonl", 'w') as f:
        f.write('\n'.join(save_text))


if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_files = os.listdir("/nlp/data/PMC/OA/jsons/") # change to your path to the json files
    # split files in chunks, 1000 files per chunk
    chunk2files = {}
    files_per_chunk = 1000

    # Split files into chunks
    for i in range(0, len(all_files), files_per_chunk):
        # Slice the list to get the next segment of files
        chunk_id = i // files_per_chunk
        chunk2files[chunk_id] = all_files[i:i + files_per_chunk]
    
    # multiprocessing
    pool = multiprocessing.Pool(processes=8)
    pool.starmap(process_files, [(files, chunk_id) for chunk_id, files in chunk2files.items()])
    pool.close()
    pool.join()