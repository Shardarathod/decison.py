import math
from collections import Counter

# --- Entropy Calculation ---
def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(data)
    return -sum((count/total) * math.log2(count/total) for count in label_counts.values())

# --- Data Splitting ---
def split_data(data, feature_index, value):
    return [row for row in data if row[feature_index] == value]

# --- Best Feature Selection ---
def best_split(data):
    base_entropy = entropy(data)
    best_info_gain = 0
    best_feature = -1
    num_features = len(data[0]) - 1

    for i in range(num_features):
        values = set(row[i] for row in data)
        new_entropy = 0
        for value in values:
            subset = split_data(data, i, value)
            prob = len(subset) / len(data)
            new_entropy += prob * entropy(subset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature

# --- Recursive Tree Building ---
def build_tree(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]

    best = best_split(data)
    best_feature = features[best]
    tree = {best_feature: {}}
    feature_values = set(row[best] for row in data)

    for value in feature_values:
        subset = split_data(data, best, value)
        new_features = features[:best] + features[best+1:]
        new_data = [row[:best] + row[best+1:] for row in subset]
        subtree = build_tree(new_data, new_features)
        tree[best_feature][value] = subtree

    return tree

# --- Prediction Function ---
def predict(tree, features, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    root_index = features.index(root)
    value = sample[root_index]
    if value not in tree[root]:
        return "Unknown"
    subtree = tree[root][value]
    return predict(subtree, features, sample)

# --- Main ---
if __name__ == "__main__":
    dataset = [
        ['Sunny', 'Hot', 'High', 'False', 'No'],
        ['Sunny', 'Hot', 'High', 'True', 'No'],
        ['Overcast', 'Hot', 'High', 'False', 'Yes'],
        ['Rain', 'Mild', 'High', 'False', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'False', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'True', 'No'],
        ['Overcast', 'Cool', 'Normal', 'True', 'Yes'],
        ['Sunny', 'Mild', 'High', 'False', 'No'],
        ['Sunny', 'Cool', 'Normal', 'False', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'False', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'True', 'Yes'],
        ['Overcast', 'Mild', 'High', 'True', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'False', 'Yes'],
        ['Rain', 'Mild', 'High', 'True', 'No'],
    ]
    features = ['Outlook', 'Temperature', 'Humidity', 'Windy']

    decision_tree = build_tree(dataset, features)
    print("📊 Decision Tree:")
    print(decision_tree)

    # Example Prediction
    test_sample = ['Sunny', 'Cool', 'High', 'True']
    prediction = predict(decision_tree, features, test_sample)
    print("\n🔍 Test Sample:", test_sample)
    print("✅ Prediction:", prediction)
