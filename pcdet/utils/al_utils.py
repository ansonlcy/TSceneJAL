import math
from collections import Counter


def calculate_category_entropy(predictions):
    """

    :param predictions: list:["Car", "Car", ''Pedestrian", "Cyclist"....]
    :return:
    """
    if len(predictions) == 0:
        return 0.0
    # 统计每个类别的数量
    category_counts = Counter(predictions)

    # 计算类别熵
    total_predictions = len(predictions)
    entropy = 0.0
    for count in category_counts.values():
        probability = count / total_predictions
        entropy -= probability * math.log2(probability)

    return entropy