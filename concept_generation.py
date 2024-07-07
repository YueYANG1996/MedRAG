import json
import copy
import openai
from tqdm import tqdm
from src.utils import RetrievalSystem
from argparse import ArgumentParser
from annotate_question import QuestionAnnotator

rag_prompt_xray = """You are an experienced radiologist. You are summarizing knowledge about 'QUERY' from chest X-rays.
Here are the documents retrieved from the corpus:

RETRIEVED_DOCUMENTS

I want you to filter and summarize the information in these documents and generate knowledge in the form of *binary questions*, e.g., "Is there lung opacity?".
Please follow instructions below strictly:
1. Those knowledge will be used to guide the diagnosis on chest X-rays, so they must be *visually identifiable* from chest X-ray only.
2. The binary questions should be concise and not too specific which can be reused for different cases.
3. The binary questions must not contain the class(disease) name, e.g., you *must not* generate "Is there cardiomegaly?" as the knowledge for "Cardiomegaly".
4. If there is not much information in the some documents, you can ignore those documents. If none of the documents contain useful information, you can skip this task by typing 'skip'.
5. Answer with the following format: question | document ID | reference sentence, e.g., Is there lung opacity? | 1234 | lung opacity is a common finding for ...

Please answer without additional information and do not add numbers or bullet points in the answer."""

rag_prompt_skin = """You are an experienced demoatologist. You are summarizing knowledge about 'QUERY' from skin lesion images.
Here are the documents retrieved from the corpus:

RETRIEVED_DOCUMENTS

I want you to filter and summarize the information in these documents and generate knowledge in the form of *binary questions*, e.g., "Is the lesion asymmetric?".
Please follow instructions below strictly:
1. Those knowledge will be used to guide the diagnosis on skin lesion images, so they must be *visually identifiable* from skin lesion images/photos only.
2. The binary questions should be concise and not too specific which can be reused for different cases.
3. The binary question must not contain the class(disease) name, e.g., you *must not* generate "Is the lesion malignant?" as the knowledge for "Malignant Lesion".
4. If there is not much information in the some documents, you can ignore those documents. If none of the documents contain useful information, you can skip this task by typing 'skip'.
5. Answer with the following format: question | document ID | reference sentence, e.g., Is the lesion asymmetric? | 1234 | asymmetric lesion is a common finding for ...

Please answer without additional information and do not add numbers or bullet points in the answer."""

prompt_xray = """You are an experienced radiologist. You are summarizing knowledge about how to diagnose CLASS_NAME from chest X-rays.
I want you to generate knowledge in the form of *binary questions*, e.g., "Is there lung opacity?".
Here are the existing binary questions in the knowledge base:
EXISTING_CONCEPTS

Please follow instructions below strictly:
1. Those knowledge will be used to guide the diagnosis on chest X-rays, so they must be *visually identifiable* from chest X-ray only.
2. The binary questions should be concise and not too specific which can be reused for different cases.
3. Each question is unique and the newly generated binary questions must be different from the existing ones.
4. The binary question must not contain the class name, e.g., you *must not* generate "Is there cardiomegaly?" for the class "Cardiomegaly".
5. Answer with the following format: question-1 | question-2 | question-3 | ..., e.g., Is there lung opacity? | Is the heart enlarged? | ...

Please answer without additional information and do not add numbers or bullet points in the answer."""

prompt_skin = """You are an experienced demoatologist. You are summarizing knowledge about how to diagnose CLASS_NAME from skin lesion images.
I want you to generate knowledge in the form of *binary questions*, e.g., "Is the lesion asymmetric?".
Here are the existing binary questions in the knowledge base:
EXISTING_CONCEPTS

Please follow instructions below strictly:
1. Those knowledge will be used to guide the diagnosis on skin lesion images, so they must be *visually identifiable* from skin lesion images/photos only.
2. The binary questions should be concise and not too specific which can be reused for different cases.
3. Each question is unique and the newly generated binary questions must *be different from* the existing ones.
4. The binary question must not contain the class name, e.g., you *must not* generate "Is the lesion malignant melanoma?" for the class "Malignant Melanoma".
5. Answer with the following format: question-1 | question-2 | question-3 | ..., e.g., Is the lesion asymmetric? | Is the lesion dark? | ...

Please answer without additional information and do not add numbers or bullet points in the answer."""


def call_openai(model_name, prompt, max_tokens=512):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt},
            ],
        max_tokens=max_tokens
        )

    return response


def check_visual(modality, concept):
    if modality == "xray": MODALITY = "chest X-ray"
    elif modality == "skin": MODALITY = "skin lesion"

    prompt = """I am evaluating the knowledge for diagnosing disease from MODALITY images. 
I want to check if the concept CONCEPT is visually identifiable from MODALITY images.
Please answer with yes or no only."""

    prompt = prompt.replace("MODALITY", MODALITY).replace("CONCEPT", concept)
    try: response = call_openai("gpt-4", prompt, 8)
    except: print("Error in prompting LLM"); return None

    answer_text = response["choices"][0]["message"]["content"].strip().lower()

    if "yes" in answer_text: return True
    elif "no" in answer_text: return False
    else: return None


def check_uniqueness(modality, concept, existing_concepts):
    if modality == "xray": MODALITY = "chest X-ray"
    elif modality == "skin": MODALITY = "skin lesion"

    prompt = """I am evaluating the knowledge for diagnosing disease from MODALITY images.
Here are the existing concepts in the knowledge base:
EXISTING

I want to add a new concept CONCEPT to the knowledge base and want to check if the new concept is different and not covered (consider paraphrasing, semantics, and synonyms) by the existing concepts.
Please answer with yes or no only."""

    prompt = prompt.replace("MODALITY", MODALITY).replace("CONCEPT", concept).replace("EXISTING", existing_concepts)
    try: response = call_openai("gpt-4", prompt, 8)
    except: print("Error in prompting LLM"); return None

    answer_text = response["choices"][0]["message"]["content"].strip().lower()

    if "yes" in answer_text: return True
    elif "no" in answer_text: return False
    else: return None


def check_class_names(modality, concept, class_names):
    if modality == "xray": MODALITY = "chest X-ray"
    elif modality == "skin": MODALITY = "skin lesion"

    prompt = """I am evaluating the knowledge for diagnosing disease from MODALITY images.
I don't want the concept to contain any class names. Here are all the class names:
CLASS_NAMES

You need to check if the concept CONCEPT contains any of the class names listed above. yes if the concept contains any class name, no otherwise
Please answer with yes or no only."""

    prompt = prompt.replace("MODALITY", MODALITY).replace("CONCEPT", concept).replace("CLASS_NAMES", ", ".join(class_names))
    try: response = call_openai("gpt-4", prompt, 8)
    except: print("Error in prompting LLM"); return None

    answer_text = response["choices"][0]["message"]["content"].strip().lower()

    if "yes" in answer_text: return False
    elif "no" in answer_text: return True
    else: return None


def check_groundability(modality, concept, question_annotator):
    question_annotator.annotate_question(concept, 100)
    yes_count, no_count = question_annotator.get_statistics("t5", modality, concept)
    if min(yes_count, no_count) / (yes_count + no_count) < 0.1: return False
    else: return True


def save_concepts(modality, corpus_name, number_of_concepts):
    bottleneck = json.load(open(f"concepts/{modality}_{corpus_name}_{number_of_concepts}.json", "r"))
    count = 0
    with open(f"../data/bottlenecks/{modality}_{corpus_name}_{number_of_concepts}.txt", "w") as f:
        for concept in bottleneck:
            f.write(concept["concept"].replace("/", "|") + "\n")
            count += 1
            if count == number_of_concepts: break


def generate_concepts_rag(prompt, id2doc, query):
    try: response = call_openai("gpt-4", prompt, 1024)
    except: print("Error in prompting LLM"); return []

    answer_text = response["choices"][0]["message"]["content"].strip()
    generated_concepts = []
    for line in answer_text.split("\n"):
        line = line.strip()
        if "|" in line:
            try:
                concept, doc_id, reference = line.split("|")
                concept = concept.replace("/", " or ").strip()
                
                if doc_id.strip() in id2doc:
                    generated_concepts.append({"concept": concept.strip(), "doc_id": doc_id.strip(), "reference_sentence": reference.strip(), "doc_content": id2doc[doc_id.strip()], "query": query})
            except:
                print("wrong format", line)
                continue

    return generated_concepts


def generate_bottleneck_rag(modality, class_names, number_of_concepts, corpus_name):
    n_of_docs_per_query = 10

    retrieval_system = RetrievalSystem(retriever_name="BM25", corpus_name=corpus_name)
    question_annotator = QuestionAnnotator(modality=modality, annotator="t5")
    
    if modality == "xray":
        retrieval_prompt = "chest X-ray QUERY diagnosis criteria"
        generate_prompt = rag_prompt_xray
    elif modality == "skin":
        retrieval_prompt = f"skin lesion QUERY diagnosis criteria"
        generate_prompt = rag_prompt_skin
    
    bottleneck = []
    used_docs = []
    queries = class_names # class names as initial queries
    max_iterations = 2000
    iteration = 0

    while len(bottleneck) < number_of_concepts:
        new_queries = []
        for query in queries:
            retrieved_documents, scores = retrieval_system.retrieve(retrieval_prompt.replace("QUERY", query), k=50, rrf_k=100)
            current_documents = []
            for doc in retrieved_documents:
                if doc["id"] not in used_docs:
                    current_documents.append(doc)
                    used_docs.append(doc["id"])

                if len(current_documents) == n_of_docs_per_query: break

            if len(current_documents) == 0: continue

            id2doc = {doc["id"]: f"ID: {doc['id']}, \n {doc['contents']} \n\n" for doc in current_documents}
            document_info = "\n".join([f"ID: {doc['id']}, \n {doc['contents']} \n\n" for doc in current_documents])
            existing_concepts = "\n".join([bottleneck[i]["concept"] for i in range(len(bottleneck))])
            prompt = generate_prompt.replace("QUERY", query).replace("RETRIEVED_DOCUMENTS", document_info)

            new_concepts = generate_concepts_rag(prompt, id2doc, query)
            for concept in new_concepts:
                if len(bottleneck) == number_of_concepts: break
                visual = check_visual(modality, concept["concept"])
                if visual != True: print("The generated concept is not visually identifiable"); continue

                if len(bottleneck) > 0: 
                    existing_concepts = "\n".join([bottleneck[i]["concept"] for i in range(len(bottleneck))])
                    unique = check_uniqueness(modality, concept["concept"], existing_concepts)
                else: unique = True
                if unique != True: print("The generated concept is already in the bottleneck"); continue

                groundable = check_groundability(modality, concept["concept"], question_annotator)

                print(concept["concept"], "Visual:", visual, "Unique:", unique, "Groundable:", groundable)

                if visual and unique and groundable:
                    bottleneck.append(concept)
                    new_queries.append(concept["concept"])
                    json.dump(bottleneck, open(f"concepts/{modality}_{corpus_name}_{number_of_concepts}.json", "w"), indent=4)
                
                if len(bottleneck) == number_of_concepts: break
        
        queries = copy.deepcopy(new_queries)
        iteration += 1
        if iteration > max_iterations: break
    
    save_concepts(modality, corpus_name, number_of_concepts)
    return bottleneck


def generate_concepts_prompt(prompt, class_name):
    try: response = call_openai("gpt-4", prompt, 1024)
    except: print("Error in prompting LLM"); return []

    answer_text = response["choices"][0]["message"]["content"].strip()
    generated_concepts = []
    
    for concept in answer_text.split("|"):
        concept = concept.strip()
        generated_concepts.append({"concept": concept, "class_name": class_name})

    return generated_concepts


def generate_bottleneck_prompt(modality, class_names, number_of_concepts):
    if modality == "xray": prompt = prompt_xray
    elif modality == "skin": prompt = prompt_skin

    question_annotator = QuestionAnnotator(modality=modality, annotator="t5")

    bottleneck = []
    while len(bottleneck) < number_of_concepts:
        for class_name in class_names:
            existing_concepts = "\n".join([bottleneck[i]["concept"] for i in range(len(bottleneck))])
            prompt = prompt.replace("CLASS_NAME", class_name).replace("EXISTING_CONCEPTS", existing_concepts)
            
            new_concepts = generate_concepts_prompt(prompt, class_name)
            for concept in new_concepts:
                if len(bottleneck) == number_of_concepts: break

                visual = check_visual(modality, concept["concept"])
                if visual != True: continue

                if len(bottleneck) > 0: 
                    existing_concepts = "\n".join([bottleneck[i]["concept"] for i in range(len(bottleneck))])
                    unique = check_uniqueness(modality, concept["concept"], existing_concepts)
                else: unique = True
                if unique != True: continue

                groundable = check_groundability(modality, concept["concept"], question_annotator)

                print(concept["concept"], "Visual:", visual, "Unique:", unique, "Groundable:", groundable)

                if visual and unique and groundable:
                    bottleneck.append(concept)
                    json.dump(bottleneck, open(f"concepts/{modality}_{corpus_name}_new_{number_of_concepts}.json", "w"), indent=4)

            json.dump(bottleneck, open(f"concepts/{modality}_{corpus_name}_new_{number_of_concepts}.json", "w"), indent=4)

            if len(bottleneck) == number_of_concepts: break
    
    save_concepts(modality, corpus_name, number_of_concepts)
    return bottleneck


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--modality", type=str, default="xray", help="Modality of the data")
    parser.add_argument("--corpus_name", type=str, default="Textbooks", help="Name of the corpus")
    parser.add_argument("--number_of_concepts", type=int, default=200, help="Number of concepts to generate")
    parser.add_argument("--openai_key", type=str, default="", help="OpenAI API key")
    args = parser.parse_args()

    openai.api_key = args.openai_key
    modality = args.modality
    corpus_name = args.corpus_name
    number_of_concepts = args.number_of_concepts

    # You can customize the class names (initial query) based on your data
    if modality == "xray": class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Edema', 'Pulmonary fibrosis', 'Pneumonia', 'Consolidation', 'Aortic enlargement', 'COVID-19', 'Pleural thickening', 'Nodule/Mass', 'Lung Opacity']
    elif modality == "skin": class_names = ["Malignant Lesion", "Benign Lesion", "Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", "Nevus", "Seborrheic Keratosis", "Solar Lentigo", "Squamous Cell Carcinoma", "Vascular lesion", "Melanocytic Nevi"]

    # this is for generating concepts without RAG
    if corpus_name == "prompt": bottleneck = generate_bottleneck_prompt(modality, class_names, number_of_concepts)
    else: bottleneck = generate_bottleneck_rag(modality, class_names, number_of_concepts, corpus_name)