import sys
import subprocess

task = sys.argv[1]
mode = sys.argv[2]
method = sys.argv[3]
sample_begin = sys.argv[4]
sample_end = sys.argv[5]
sample_step = sys.argv[6]

if task == "NLI":
    process = subprocess.Popen(["python", "./test-task-after-code/test-NLI-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "OpenDomainQA":
    process = subprocess.Popen(["python", "./test-task-after-code/test-OpenDomainQA-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "SentimentAnalysis":
    process = subprocess.Popen(["python", "./test-task-after-code/test-SentimentAnalysis-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "summarization":
    process = subprocess.Popen(["python", "./test-task-after-code/test-summarization-after.py", mode, method, sample_begin, sample_end, sample_step])

process.wait()
print("Done")