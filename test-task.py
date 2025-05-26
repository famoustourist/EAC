import sys
import subprocess

task = sys.argv[1]


if task == "NLI":
    process = subprocess.Popen(["python", "./test-task-code/test-NLI.py"])

if task == "OpenDomainQA":
    process = subprocess.Popen(["python", "./test-task-code/test-OpenDomainQA.py"])

if task == "SentimentAnalysis":
    process = subprocess.Popen(["python", "./test-task-code/test-SentimentAnalysis.py"])

if task == "summarization":
    process = subprocess.Popen(["python", "./test-task-code/test-summarization.py"])

process.wait()
print("Done")