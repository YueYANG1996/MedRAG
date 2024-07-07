import os
import time
import pickle
import random
import openai
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
from argparse import ArgumentParser
import torch
from torch.nn.functional import softmax
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

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


class QuestionAnnotator():
    def __init__(self, modality, annotator):
        self.modality = modality
        self.annotator = annotator

        if self.modality == "xray":
            self.finding2report_ids = json.load(open("clinical_data/MIMIC-CXR/finding2report_ids.json", "r"))
            self.finding_embeds = pickle.load(open("clinical_data/MIMIC-CXR/finding_embeds.pkl", "rb"))
            self.all_reports = json.load(open("clinical_data/MIMIC-CXR/all_reports.json", "r"))
        
        elif self.modality == "skin":
            self.finding2report_ids = json.load(open("clinical_data/ISIC/finding2report_ids.json", "r"))
            self.finding_embeds = pickle.load(open("clinical_data/ISIC/finding_embeds.pkl", "rb"))
            self.all_reports = json.load(open("clinical_data/ISIC/all_reports.json", "r"))
        
        self.all_findings = list(self.finding2report_ids.keys())

        self.report_id2findings = {}
        for finding, report_ids in tqdm(self.finding2report_ids.items()):
            for report_id in report_ids:
                if report_id not in self.report_id2findings:
                    self.report_id2findings[report_id] = []
                self.report_id2findings[report_id].append(finding)

        
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2', device = device)

        if annotator == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", low_cpu_mem_usage=True)

            self.yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
            self.no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]


    def score_reports(self, query):
        query_embed = self.sbert_model.encode([query], batch_size = 64, show_progress_bar=True)
        cos_scores = util.pytorch_cos_sim(query_embed, self.finding_embeds)[0]
        cos_scores = cos_scores.cpu()

        finding2score = {}
        for i in range(len(self.all_findings)):
            finding2score[self.all_findings[i]] = cos_scores[i].item()

        report2scores = {}
        for report_id, findings in self.report_id2findings.items():
            scores = []
            for finding in findings:
                scores.append(finding2score[finding])
            report2scores[report_id] = np.max(scores)
        
        sorted_reports = [(k, v) for k, v in sorted(report2scores.items(), key=lambda item: item[1], reverse=True)]

        return sorted_reports
    

    def annotate_question(self, question, number_of_reports):
        if self.annotator == "gpt4": self.annotate_question_gpt4(question, number_of_reports)
        elif self.annotator == "t5": self.annotate_question_t5(question, number_of_reports)


    def answer_question_gpt4(self, question, report_id, report):
        if self.modality == "xray": modality_name = "chest X-ray"
        elif self.modality == "skin": modality_name = "skin lesion"

        prompt = f"""Here is a {modality_name} report:
{report}

Task: Answer the following question based on the above report and your medical knowledge.
Guide: Please answer with yes or no only. If the report does not explictly contains the information, please infer from your medical knowledge.
Question: {question}
Choices: yes, no.
Answer: """
        
        try:
            response = call_openai("gpt-4", prompt, max_tokens=8)
        except:
            print("Error: openai api call failed")
            return "invalid"

        answer_text = response["choices"][0]["message"]["content"].strip().lower()

        if "yes" in answer_text:
            answer = "yes"
        elif "no" in answer_text:
            answer = "no"
        else:
            answer = "invalid"
        
        if answer != "invalid":
            with open(f"../data/concept_annotation_{self.modality}/annotations_gpt4/{question}/{report_id}.txt", "w") as f:
                f.write(answer)
                
        return answer


    def annotate_question_gpt4(self, question, number_of_reports):
        number_of_reports_per_class = number_of_reports // 2

        save_dir = f"../data/concept_annotation_{self.modality}/annotations_gpt4/{question}"
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        done_report_ids = [f.split(".")[0] for f in os.listdir(save_dir)]
        relevant_report_ids = [report_id for report_id, _ in self.score_reports(question)[:number_of_reports_per_class]]
        done_relevant_report_ids = list(set(relevant_report_ids) & set(done_report_ids))

        number_of_negative_reports = number_of_reports_per_class - len(done_relevant_report_ids)
        random.seed(0)
        irrelevant_report_ids = [report_id for report_id, _ in random.sample(self.score_reports(question)[number_of_reports_per_class:], number_of_negative_reports)]
        all_report_ids = list(set(relevant_report_ids + irrelevant_report_ids))
        
        rest_report_ids = list(set(all_report_ids) - set(done_report_ids))
        print("Number of reports to annotate: ", len(rest_report_ids))

        if len(rest_report_ids) == 0: return
        
        start_time = time.time()
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(self.answer_question_gpt4, [(question, report_id, self.all_reports[report_id]) for report_id in rest_report_ids])
        pool.close()
        pool.join()
        end_time = time.time()

        print("Time used:", end_time - start_time)

    
    def generate_and_get_probabilities(self, prompts, max_new_tokens=8):
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        generated_outputs = self.t5_model.generate(**model_inputs, 
                                            max_new_tokens=max_new_tokens, 
                                            output_scores=True, 
                                            return_dict_in_generate=True)
        
        # Extract the logits of the first token of the generated sequence
        all_logits = generated_outputs.scores[0]

        outputs = []
        for i in range(len(prompts)):
            logits = all_logits[i]

            # get yes and no logits
            yes_logits = logits[self.yes_token_id]
            no_logits = logits[self.no_token_id]

            # Convert logits to probabilities
            probs = softmax(torch.stack([yes_logits, no_logits]), dim=-1)

            output = {"text": self.tokenizer.batch_decode(generated_outputs.sequences)[i],
                    "probabilities": {"yes": probs[0].item(), "no": probs[1].item()},
                    "logits": {"yes": yes_logits.item(), "no": no_logits.item()}}
            
            outputs.append(output)

        return outputs


    def annotate_question_t5(self, question, number_of_reports):
        if self.modality == "xray": modality_name = "chest x-ray"
        elif self.modality == "skin": modality_name = "skin lesion"

        prompt_template = f"""Here is a {modality_name} report:
REPORT

Task: Answer the following question based on the above report with "Yes" or "No":
Question: QUESTION
Answer: """

        save_dir = f"../data/concept_annotation_{self.modality}/annotations_t5/{question}"
        output_dir = f"../data/concept_annotation_{self.modality}/annotations_t5_outputs/{question}"

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        number_of_reports_per_class = number_of_reports // 2

        done_report_ids = [f.split(".")[0] for f in os.listdir(save_dir)]
        relevant_report_ids = [report_id for report_id, _ in self.score_reports(question)[:number_of_reports_per_class]]
        done_relevant_report_ids = list(set(relevant_report_ids) & set(done_report_ids))

        number_of_negative_reports = number_of_reports_per_class - len(done_relevant_report_ids)
        random.seed(0)
        irrelevant_report_ids = [report_id for report_id, _ in random.sample(self.score_reports(question)[number_of_reports_per_class:], number_of_negative_reports)]
        all_report_ids = list(set(relevant_report_ids + irrelevant_report_ids))
        
        rest_report_ids = list(set(all_report_ids) - set(done_report_ids))
        print("Number of reports to annotate: ", len(rest_report_ids))

        if len(rest_report_ids) == 0: return

        batch_size = 4
        report_batches = [rest_report_ids[i:i + batch_size] for i in range(0, len(rest_report_ids), batch_size)]

        start_time = time.time()

        for batch in tqdm(report_batches):
            prompts = []
            for report_id in batch:
                report = self.all_reports[report_id]
                # only keep the first 400 tokens of the report
                report = " ".join(report.split()[:400])

                prompt = prompt_template.replace("REPORT", report).replace("QUESTION", question)
                prompts.append(prompt)

            outputs = self.generate_and_get_probabilities(prompts)
            for i in range(len(batch)):
                report_id = batch[i]
                with open(f"{save_dir}/{report_id}.txt", "w") as f:
                    if outputs[i]["probabilities"]["yes"] > outputs[i]["probabilities"]["no"]:
                        f.write("yes")
                    else:
                        f.write("no")
                    
                with open(f"{output_dir}/{report_id}.json", "w") as f:
                    json.dump(outputs[i], f)
        
        end_time = time.time()
        print("Time used:", end_time - start_time)


    def get_statistics(self, annotator, modality, question):
        save_dir = f"../data/concept_annotation_{modality}/annotations_{annotator}/{question}"
        annotation_files = os.listdir(save_dir)

        yes_count = 0
        no_count = 0
        for annotation_file in annotation_files:
            with open(f"{save_dir}/{annotation_file}", "r") as f:
                answer = f.read().strip()
            if answer == "yes":
                yes_count += 1
            elif answer == "no":
                no_count += 1
            else:
                print("Error: invalid answer")
        
        print("Number of yes answers:", yes_count)
        print("Number of no answers:", no_count)

        return yes_count, no_count


def annotate_reports(annotator, modality, questions, number_of_reports):    
    question_annotator = QuestionAnnotator(modality, annotator)
    
    for question in questions:
        print("Question:", question)

        question_dir = f"../data/concept_annotation_{modality}/annotations_{annotator}/{question}"
        if not os.path.exists(question_dir): os.makedirs(question_dir)

        number_of_done_reports = len(os.listdir(question_dir))

        if number_of_done_reports >= number_of_reports:
            print("already annotated")
            continue
        elif number_of_done_reports < 100:
            # test annotation of 100 reports to check if we can find enough relevant reports
            question_annotator.annotate_question(question, 100)
            print("Test annotation of 100 reports")

        yes_count, no_count = question_annotator.get_statistics(annotator, modality, question)

        if min(yes_count, no_count) / (yes_count + no_count) < 0.1:
            print("Not enough relevant reports, ignore this question")
            continue
        else:
            question_annotator.annotate_question(question, number_of_reports)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--annotator", type=str, default="gpt4", help="Annotator to use, gpt4 or t5")
    parser.add_argument("--modality", type=str, default="xray", help="Modality of the data")
    parser.add_argument("--bottleneck_name", type=str, default="PubMed", help="Bottleneck to use")
    parser.add_argument("--number_of_reports", type=int, default=1000, help="Number of reports to annotate for each question/concept")
    parser.add_argument("--openai_key", type=str, default="", help="OpenAI API key")
    args = parser.parse_args()

    openai.api_key = args.openai_key

    if args.annotator == "t5":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    with open(f"../data/bottlenecks/{args.bottleneck_name}.txt", "r") as f:
        questions = f.readlines()
        questions = [q.strip() for q in questions]

    annotate_reports(args.annotator, args.modality, questions, args.number_of_reports)