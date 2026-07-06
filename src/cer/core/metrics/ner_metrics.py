from typing import List, Literal, Tuple, Union

import numpy as np

from cer.core.interface import Entity

# =====================================
# Helper functions
# =====================================


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        The precision, recall, and F1 score.
    """
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1


def _longest_common_substring(text1: str, text2: str) -> float:
    """Find the longest common substring between two strings.

    Args:
        text1: The first string.
        text2: The second string.

    Returns:
        The length of the longest common substring.
    """

    m = len(text1)
    n = len(text2)
    lcsuff = np.zeros((m + 1, n + 1))
    result = 0  # To store length of the longest common substring

    # Building the lcsuff table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcsuff[i][j] = 0
            elif text1[i - 1] == text2[j - 1]:
                lcsuff[i][j] = lcsuff[i - 1][j - 1] + 1
                result = max(result, lcsuff[i][j])
            else:
                lcsuff[i][j] = 0

    return result


# ======================================
# NER evaluation functions
# ======================================


def _exact_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity], label: str = None
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using exact matching.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.
        label: The label based on which we want to evaluate predictions.
            If None, uses all labels in the evaluation (default: None).

    Returns:
        The true positives, false positives, and false negatives.
    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents if label is None or ent["label"] == label)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents if label is None or ent["label"] == label)

    if len(true_ents_set) == 0 and len(pred_ents_set) == 0:
        return 0, 0, 0

    tp = len(true_ents_set & pred_ents_set)
    fp = len(pred_ents_set - true_ents_set)
    fn = len(true_ents_set - pred_ents_set)

    return tp, fp, fn


def _relaxed_ner_evaluation(
    true_ents: List[Entity],
    pred_ents: List[Entity],
    label: str = None,
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using relaxed matching.

    When using relaxed matching, the algorithm considers an entity as correct if the
    predicted entity contains the true entity. The labels of both entities must also match.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.
        label: The label based on which we want to evaluate predictions.
            If None, uses all labels in the evaluation (default: None).

    Returns:
        The true positives, false positives, and false negatives.
    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents if label is None or ent["label"] == label)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents if label is None or ent["label"] == label)

    if len(true_ents_set) == 0 and len(pred_ents_set) == 0:
        return 0, 0, 0

    tp = 0
    for true_ent in true_ents_set:
        for pred_ent in pred_ents_set:
            if true_ent[1] == pred_ent[1] and true_ent[0] in pred_ent[0]:
                tp += 1
                break

    fp = max(len(pred_ents_set) - tp, 0)
    fn = max(len(true_ents_set) - tp, 0)
    return tp, fp, fn


def _overlap_ner_evaluation(
    true_ents: List[Entity],
    pred_ents: List[Entity],
    label: str = None,
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using overlap matching.

    When using overlap matching, the algorithm considers an entity as correct if there
    is an overlap between the true and predicted entities. The labels of both entities must also match.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.
        label: The label based on which we want to evaluate predictions.
            If None, uses all labels in the evaluation (default: None).

    Returns:
        The true positives, false positives, and false negatives.

    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents if label is None or ent["label"] == label)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents if label is None or ent["label"] == label)

    if len(true_ents_set) == 0 and len(pred_ents_set) == 0:
        return 0, 0, 0

    tp = 0
    for true_ent in true_ents_set:
        for pred_ent in pred_ents_set:
            if true_ent[1] == pred_ent[1] and _longest_common_substring(true_ent[0], pred_ent[0]) > 0:
                tp += 1
                break

    fp = max(len(pred_ents_set) - tp, 0)
    fn = max(len(true_ents_set) - tp, 0)
    return tp, fp, fn


def _bertscore_ner_evaluation(
    true_ents: List[Entity],
    pred_ents: List[Entity],
    label: str = None,
) -> Tuple[float, float, float, int]:
    """Evaluate named entity recognition performance using bertscore matching.

    It is based on the longest common substring between the true and predicted entities.
    Furthermore, the label of the entity is also considered during the evaluation. The
    inspiration for this algorithm comes from the following paper:

    @inproceedings{bert-score,
        title={BERTScore: Evaluating Text Generation with BERT},
        author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=SkeHuCVFDr}
    }

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.
        label: The label based on which we want to evaluate predictions.
            If None, uses all labels in the evaluation (default: None).

    Returns:
        The precision, recall, and F1 score, and the number indicating if the pair was considered for the evaluation.
    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents if label is None or ent["label"] == label)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents if label is None or ent["label"] == label)

    if len(true_ents_set) == 0 and len(pred_ents_set) == 0:
        return 0.0, 0.0, 0.0, 0
    if len(true_ents_set) == 0 or len(pred_ents_set) == 0:
        return 0.0, 0.0, 0.0, 1

    # get label matching score
    label_matrix = np.zeros((len(true_ents_set), len(pred_ents_set)))
    text_matrix = np.zeros((len(true_ents_set), len(pred_ents_set)))
    for i, true_ent in enumerate(true_ents_set):
        for j, pred_ent in enumerate(pred_ents_set):
            label_matrix[i, j] = true_ent[1] == pred_ent[1]
            text_matrix[i, j] = _longest_common_substring(true_ent[0], pred_ent[0])

    matrix = text_matrix * label_matrix
    p = np.mean(np.max(matrix, axis=0))
    r = np.mean(np.max(matrix, axis=1))
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1, 1


class NERMetrics:

    def __init__(self, metrics: List[Literal["exact", "relaxed", "overlap", "bertscore"]]):
        self.metrics = metrics

    def evaluate_ner_performance(
        self,
        true_ents: List[List[Entity]],
        pred_ents: List[List[Entity]],
        match_type: Union[Literal["exact"], Literal["relaxed"], Literal["overlap"], Literal["bertscore"]] = "exact",
        label: str = None,
    ) -> Tuple[float, float, float]:
        """Evaluate named entity recognition performance.

        Args:
            true_ents: List of true entities for each example.
            pred_ents: List of predicted entities for each example.
            match_type: The evaluation method to use, either "exact", "relaxed", "overlap" or "bertscore".
            label: The label based on which we want to evaluate predictions.
                If None, uses all labels in the evaluation (default: None).

        Returns:
            The precision, recall, and F1 score.
        """
        if len(true_ents) != len(pred_ents):
            raise ValueError("The number of true and predicted entities must be the same.")

        if match_type not in ["exact", "relaxed", "overlap", "bertscore"]:
            raise ValueError(f"Unknown match_type method: {match_type}")

        if match_type == "bertscore":
            p, r, f1, n = 0.0, 0.0, 0.0, 0
            for true_ent, pred_ent in zip(true_ents, pred_ents):
                _p, _r, _f1, _n = _bertscore_ner_evaluation(true_ent, pred_ent, label=label)
                p, r, f1, n = p + _p, r + _r, f1 + _f1, n + _n
            if n == 0:
                return 0.0, 0.0, 0.0
            p, r, f1 = p / n, r / n, f1 / n
            return p, r, f1

        if match_type == "exact":
            eval_func = _exact_ner_evaluation
        elif match_type == "relaxed":
            eval_func = _relaxed_ner_evaluation
        elif match_type == "overlap":
            eval_func = _overlap_ner_evaluation

        tp, fp, fn = 0, 0, 0
        for true_ent, pred_ent in zip(true_ents, pred_ents):
            _tp, _fp, _fn = eval_func(true_ent, pred_ent, label=label)
            tp, fp, fn = tp + _tp, fp + _fp, fn + _fn

        p, r, f1 = _compute_precision_recall_f1(tp, fp, fn)
        return p, r, f1
