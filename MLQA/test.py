import os
import subprocess

context_language = 'en'
question_language = 'en'

print(os.getcwd())

with open(f"./evaluation_output_{context_language}_{question_language}_baseline.txt", "w") as f:
    subprocess.run(
        ["python", "mlqa_evaluation_v1.py", f"./Data/test/test-context-{context_language}-question-{question_language}.json", f"./formatted_predictions_{context_language}_{question_language}_baseline.json", f"{context_language}"],
        stdout=f,
    )