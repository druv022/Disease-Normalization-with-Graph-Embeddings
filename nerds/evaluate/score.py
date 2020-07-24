

def annotation_precision_recall_f1score(y_pred, y_true,
                                        ann_type="annotation",
                                        ann_label=None,
                                        ann_value=None):
    """ Calculates precision recall and F1-score metrics.

        Args:
            y_pred (list(AnnotatedDocument)): The predictions of an NER
                model in the form of a list of annotated documents.
            y_true (list(AnnotatedDocument)): The ground truth set of
                annotated documents.
            ann_type (str, optional): The type of annotations used to
                evaluate predictions. The default value is "annotation" which
                means only named entities are counted. Other values are
                "relation" (to evaluate relations) and "norm" (to evaluate
                normalizations).
            ann_label (str | list(str), optional): The entity label(s) for which
                the scores are calculated. It defaults to None, which means
                all annotated entities.
            ann_value (str | list(str), optional): The entity value(s) for which
                the scores are calculated. It defaults to None, which means
                all values of entities.

        Returns:
            (3-tuple(float)): (Precision, Recall, F1-score)
    """
    # Flatten all annotations
    all_y_pred_ann = []
    all_y_true_ann = []
    for annotated_document in y_pred:
        all_y_pred_ann.extend(annotated_document.annotations_by_value(
            ann_value=ann_value,
            ann_label=ann_label,
            ann_type=ann_type
        ))
    for annotated_document in y_true:
        all_y_true_ann.extend(annotated_document.annotations_by_value(
            ann_value=ann_value,
            ann_label=ann_label,
            ann_type=ann_type
        ))
    return precision_recall_f1score(all_y_pred_ann, all_y_true_ann)


def precision_recall_f1score(y_pred, y_true):
    """ Calculates precision recall and F1-score metrics for the predicted set compared to the ground truth set.
        True positives (TP) are predictions that are in the ground truth set,
        false positives (FP) are predictions that are not in the ground truth set,
        false negatives are (FN) items that are in the ground truth set but not in the predicted set.
    Args:
        y_pred (list): predicted set
        y_true (list): ground truth set
    Returns:
        (3-tuple(float)): (Precision, Recall, F1-score)
    """
    # assume perfect accuracy if given empty sets
    if not y_pred and not y_true:
        return 1.0, 1.0, 1.0
    # count TP, FP, FN
    tp = 0.0
    fp = 0.0
    fn = 0.0
    y_pred_lookup = set(y_pred)
    y_true_lookup = set(y_true)
    for item in y_pred:
        if item in y_true_lookup:
            tp += 1.0
        else:
            fp += 1.0
    for item in y_true:
        if item not in y_pred_lookup:
            fn += 1.0

    # calculate precision, recall, f1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
    f1_score = (2 * precision * recall) / (precision + recall) if \
        (precision + recall) > 0 else 0.

    return precision, recall, f1_score
