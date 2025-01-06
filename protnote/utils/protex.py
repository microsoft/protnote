# MOST CONTENT Directly copied from https://github.com/google-deepmind/protex/tree/main
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Common utilities for evaluation."""

import collections
import numpy as np
from google.cloud import storage
import json
from sklearn.metrics import precision_recall_curve

BUCKET_NAME = "protex"
PREDICTIONS_DIR = "predictions"
NEG_INF = -1e9

def read_jsonl_from_gcs(bucket_name, file_name):
    """Reads a JSONL file from a public Google Cloud Storage bucket."""

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    data = blob.download_as_bytes()
    lines = data.decode('utf-8').splitlines()

    json_objects = []
    for line in lines:
        json_objects.append(json.loads(line))
    return json_objects


def preprocess_preds(dataset, scores, all_labels):
  """Return predictions and ground truth labels in common format."""
  # Map of accession to map of label to score.
  accession_to_predictions = collections.defaultdict(dict)
  for row in scores:
    score = float(row["score"])
    accession = row["inputs"]["accession"]
    label = row["inputs"]["label"]
    accession_to_predictions[accession][label] = score

  true_labels = []
  pred_scores = []
  for row in dataset:
    accession = row["accession"]
    predictions = accession_to_predictions[accession]
    gold_labels = set(row["labels"])
    true_labels_row = []
    pred_scores_row = []
    for label in all_labels:
      true_label = 1 if label in gold_labels else 0
      pred_score = predictions.get(label, NEG_INF)
      true_labels_row.append(true_label)
      pred_scores_row.append(pred_score)
    true_labels.append(true_labels_row)
    pred_scores.append(pred_scores_row)

  return np.array(true_labels), np.array(pred_scores)


def get_all_labels(dataset, predictions):
  """Return union of labels in predictions and dataset."""
  all_labels = set()
  for row in dataset:
    for label in row["labels"]:
      all_labels.add(label)
  for row in predictions:
    all_labels.add(row["inputs"]["label"])
  return all_labels


def read_jsonl(filepath, verbose=True):
  """Read jsonl file to a List of Dicts."""
  data = []
  with open(filepath, "r") as jsonl_file:
    for idx, line in enumerate(jsonl_file):
      if verbose and idx % 1000 == 0:
        # Print the index every 1000 lines.
        print("Processing line %s." % idx)
      try:
        data.append(json.loads(line))
      except json.JSONDecodeError as e:
        print("Failed to parse line: `%s`" % line)
        raise e
  if verbose:
    print("Loaded %s lines from %s." % (len(data), filepath))
  return data


def get_max_f1(true_labels, pred_scores):
  """Return maximum micro-averaged F1 score."""
  true_labels = true_labels.flatten()
  pred_scores = pred_scores.flatten()
  precisions, recalls, thresholds = precision_recall_curve(
      true_labels, pred_scores
  )
  # The last values have no associated threshold.
  precisions = precisions[:-1]
  recalls = recalls[:-1]

  f1_scores = 2 * precisions * recalls / (precisions + recalls)
  max_f1_score_idx = np.argmax(f1_scores)
  max_threshold = thresholds[max_f1_score_idx]
  max_f1 = f1_scores[max_f1_score_idx]
  print(f"max_threshold: {max_threshold}")
  return max_f1