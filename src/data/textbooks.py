import os
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fdir = "corpus/textbooks/en"
    fnames = sorted(os.listdir(fdir))

    if not os.path.exists("corpus/textbooks/chunk"):
        os.makedirs("corpus/textbooks/chunk")
        
    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join("corpus/textbooks/en", fname)
        texts = text_splitter.split_text(open(fpath).read().strip())
        saved_text = [json.dumps({"title": fname.strip(".txt"), "content": re.sub("\s+", " ", texts[i])}) for i in range(len(texts))]
        with open("corpus/textbooks/chunk/{:s}".format(fname.replace(".txt", ".jsonl")), 'w') as f:
            f.write('\n'.join(saved_text))