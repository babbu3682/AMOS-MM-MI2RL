import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from LaMed.src.dataset.multi_dataset import CapDataset, CapDataset_TEST
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
# from green_metric_full import calculate_mean_green_score
from green_metric_split import calculate_mean_green_score


accuracy  = evaluate.load("accuracy")
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor    = evaluate.load("meteor")
rouge     = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/workspace/0.Challenge/MICCAI2024_AMOSMM/M3D/LaMed/output/LaMed-finetune-0000")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--proj_out_num', type=int, default=256)

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--cap_data_val_path', type=str, default="/workspace/0.Challenge/MICCAI2024_AMOSMM/dataset/AMOS-MM/valid_cap_final.csv")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-finetune-0000/eval_caption/")
    parser.add_argument('--cap_data_test_path', type=str, default="/workspace/0.Challenge/MICCAI2024_AMOSMM/dataset/AMOS-MM/test_cap.csv")
    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=300)

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds  = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels
        

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.eos_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        trust_remote_code=True
    )
    model = model.to(device=device)

    if args.inference:
        test_dataset = CapDataset_TEST(args, tokenizer=tokenizer, mode='test')
    else:
        test_dataset = CapDataset(args, tokenizer=tokenizer, mode='valid') # test1k

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        if args.inference:
            writer.writerow(["image_path", "region", "Question", "Pred"])
            for sample in tqdm(test_dataloader):
                question = sample["question"]

                text_tensor = tokenizer(question, return_tensors="pt")

                image          = sample["image"].to(device=device)
                input_id       = text_tensor['input_ids'].to(device=device)
                attention_mask = text_tensor["attention_mask"].to(device=device)

                generation = model.generate(image, input_id, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                decoded_preds = [pred.strip() for pred in generated_texts]

                writer.writerow([sample["path"][0], sample["region"][0], question[0], decoded_preds[0]])

        else:
            writer.writerow(["image_path", "region", "Question", "Ground Truth", "Pred", "bleu", "rouge1", "meteor", "bert_f1", "green_score", "explanations"])        
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer   = sample['answer']

                text_tensor = tokenizer(question, return_tensors="pt")

                image          = sample["image"].to(device=device)
                input_id       = text_tensor['input_ids'].to(device=device)
                attention_mask = text_tensor["attention_mask"].to(device=device)

                generation = model.generate(image, input_id, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                
                result = dict()
                decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
                
                try:
                    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                    result["bleu"] = bleu_score['bleu']
                except:
                    result["bleu"] = 0.0
                    print("image_path == ", sample["path"][0])
                    print("decoded_preds == ", decoded_preds)
                    print("decoded_labels == ", decoded_labels)

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                green_score, explanations = calculate_mean_green_score(pred_report=decoded_preds, gt_report=decoded_labels)
                result["green_score"] = green_score

                writer.writerow([sample["path"][0], sample["region"][0], question[0], answer[0], decoded_preds[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"], result["green_score"], explanations[0]])


if __name__ == "__main__":
    main()
       