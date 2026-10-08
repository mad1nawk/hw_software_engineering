# распознавание языка

from transformers import pipeline
import os

os.environ["HF_TOKEN"] = "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"

classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
)

text = "Hello"
result = classifier(text)
print(result)
