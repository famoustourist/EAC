import os
import sys
import csv
import json
from easyeditor import BaseEditor
from transformers import GPT2Tokenizer
from easyeditor import ROMEHyperParams
from easyeditor import MEMITHyperParams
from easyeditor import KNHyperParams
from easyeditor import MENDHyperParams



tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

test_data = json.load(open(os.path.join("./data/edited-data", 'zsre.json'), 'r', encoding='utf-8'))
data = []
for i in range(3111):
    j = i % 1037
    data.append(test_data[j])

prompts = []
ground_truth = []
target_new = []
subject = []

mode = sys.argv[1]
method = sys.argv[2]
sample_begin = sys.argv[3]
sample_end = sys.argv[4]
sample_step = sys.argv[5]
sample_total = (int(sample_begin) - int(sample_end))//int(sample_step)

for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        prompts.append(data[i]['src'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        ground_truth.append(data[i]['pred'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        target_new.append(data[i]['alt'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        subject.append(data[i]['subject'])


if mode == "Instance-Sequential" and method == "ROME-EAC":
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )

if mode == "Instance-Sequential" and method == "MEMIT-EAC":
    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )


with open('./data/task-data/test-SentimentAnalysis.tsv') as f:
    text = []
    label = []
    generation_prompts_list = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    others = 0
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        text.append(line[0])
        label.append(line[1])
    for i in range(1,len(text)):
        generation_prompts = [f"For each snippet of text,label the sentiment of the text as positive or negative.The answer should be exact 'positive' or 'negative'. text: {text[i]} answer:"]
        generation_prompts_list.append(generation_prompts)
    for j in range(len(generation_prompts_list)):
        result = open(f"./test-result/test-SentimentAnalysis/result-SentimentAnalysis-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
        batch = tokenizer(generation_prompts_list[j], return_tensors='pt', padding="max_length")

        post_edit_outputs = edited_model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=1)

        Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("answer:")[-1]
        result.write(str(label[j+1]) + '\t')
        result.write(str(predict) + '\n')
        if ('positive' in predict.lower()) or ('negative' in predict.lower()):
            if ('positive' in predict.lower()) and (int(label[j+1]) == 1):
                TP = TP + 1
            elif ('negative' in predict.lower()) and (int(label[j+1]) == 0):
                FN = FN + 1
            elif ('negative' in predict.lower()) and (int(label[j+1]) == 1):
                TN = TN + 1
            elif ('positive' in predict.lower()) and (int(label[j+1]) == 0):
                FP = FP + 1
        else:
            others = others + 1
        result.close()


result = open(f"./test-result/test-SentimentAnalysis/result-SentimentAnalysis-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")

if others == 872:
    result.write("error" + '\n')
else:
    accuracy = (TP + FN)/(TP + FN + TN + FP)
    total_accuracy = (TP + FN)/(TP + FN + TN + FP + others)
    result.write(str(TP) + '\t')
    result.write(str(FN) + '\t')
    result.write(str(TN) + '\t')
    result.write(str(FP) + '\t')
    result.write(str(others) + '\n')
    result.write(str(accuracy) + '\t')
    result.write(str(total_accuracy) + '\n')
result.close()
