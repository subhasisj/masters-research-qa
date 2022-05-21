import json
import subprocess  # to read json
import warnings  # to ignore warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datasets import Dataset, get_dataset_config_names, load_dataset

warnings.filterwarnings("ignore")
import collections
import os

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoAdapterModel,
    AutoTokenizer,
    Trainer,
    AdapterConfig,
)
from transformers.adapters.composition import Stack

context_language = "hi"
batch_size = 16

# model = AutoModelForQuestionAnswering.from_pretrained(
#     f"subhasisj/{context_language}-kd-XLM-minilmv2-4",
# )
model_path = f"../multi-task-learning/{context_language}-adapter-en-trained"
model = AutoAdapterModel.from_pretrained(
    model_path
)

config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
adapter_name_1 = model.load_adapter("en", config=config,model_name = model_path)
adapter_name_3 = model.load_adapter("hi", config=config)
adapter_name_3 = model.load_adapter("pfeiffer_xlm_base", config=config)
model.set_active_adapters([adapter_name_1, adapter_name_3])



# adapter_name_1 = model.load_adapter(
#     os.path.join(model_path,"en"), config=config, model_name="xlm-roberta-base"
# )
# adapter_name_2 = model.load_adapter(
#     os.path.join(model_path,"hi"), config=config
# )
# task_adapter = model.load_adapter(
#    os.path.join(model_path,"pfeiffer_xlm_base"), source="hf", load_as="pfeiffer_xlm_base"
# )
# model.active_adapters = Stack(adapter_name_2, task_adapter)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

device = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Models Loaded to {device}....")
print()
# Load model into Huggingface Trainer

trainer = Trainer(model=model)
pad_on_right = tokenizer.padding_side == "right"
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128  # Th
max_answer_length = 30

# ## MLQA Format
#
# `{split}-context-{context_language}-question-{question_language}.json`
#

# %%
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or offset_mapping[start_index] == []
                        or offset_mapping[end_index] == []
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        # if not squad_v2:
        # predictions[example["id"]] = best_answer["text"]
        # else:
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions


available_languages = ["en", "hi", "es", "zh", "ar", "de", "vi"]
# available_languages = ["vi"]

# create a directory for context_language
if not os.path.exists(context_language):
    os.mkdir(context_language)

# create a directory called TAPT inside the context_language directory
technique = "multi_task"
if not os.path.exists(f"{context_language}/{technique}"):
    os.mkdir(f"{context_language}/{technique}")

for language in available_languages:
    question_language = language
    print(f"Evaluating {question_language} questions in {context_language} context.")
    mlqa = load_dataset("mlqa", f"mlqa.{context_language}.{question_language}")

    validation_features = mlqa.map(
        prepare_validation_features,
        batch_size=batch_size,
        batched=True,
        remove_columns=mlqa["test"].column_names,
    )

    raw_predictions = trainer.predict(validation_features["test"])

    validation_features["test"].set_format(
        type=validation_features["test"].format["type"],
        columns=list(validation_features["test"].features.keys()),
    )

    examples = mlqa["test"]
    features = validation_features["test"]

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    final_predictions = postprocess_qa_predictions(
        mlqa["test"], validation_features["test"], raw_predictions.predictions
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in final_predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in mlqa["test"]]

    # for each item in formatted predictions create a dictionary with the id as key and the prediction_text as value
    prediction_dict = {p["id"]: p["prediction_text"] for p in formatted_predictions}

    predictions_file = f"./{context_language}/{technique}/formatted_predictions_{context_language}_{question_language}_kd.json"
    with open(
        predictions_file,
        "w",
    ) as f:
        json.dump(prediction_dict, f)

    # Execute mlqa_evaluation_v1 script with arguments for dataset_file file and prediction_file  and write the console output to a file
    evaluation_output_path = f"./{context_language}/{technique}/evaluation_output_{context_language}_{question_language}_kd.txt"
    with open(evaluation_output_path, "w") as f:
        subprocess.run(
            [
                "python",
                "mlqa_evaluation_v1.py",
                f"./Data/test/test-context-{context_language}-question-{question_language}.json",
                predictions_file,
                f"{context_language}",
            ],
            stdout=f,
        )
    print(f"Evaluation output written to file: {evaluation_output_path}")
# os.system(
#     f"python mlqa_evaluation_v1.py ./MLQA/Data/test/test-context-{context_language}-question-{question_language}.json ./MLQA/formatted_predictions_{context_language}_{question_language}_baseline.json {context_language}"
# )

# %% [markdown]
# ## Run MLQA Evaluation using:
# -   ARG1: Test Data Path
# -   ARG2: Predictions Path
# -   ARG3: Answer Language
#
# `python MLQA/mlqa_evaluation_v1.py ./MLQA/Data/test/test-context-en-question-hi.json ./MLQA/formatted_predictions_en_hi_baseline.json en`
