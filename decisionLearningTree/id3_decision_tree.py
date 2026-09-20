import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np

#  --- Read CSV ---
descs, ratings = [], []
with open("car.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f) # uses the header row as keys
    for row in reader:
        ratings.append(row["class"])
        descs.append({
            "buying": row["buying"],
            "maint": row["maint"],
            "doors": row["doors"],
            "persons": row["persons"],
            "lug_boot": row["lug_boot"],
            "safety": row["safety"],
            })
attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
classes = ["unacc", "acc", "good", "vgood"]
# descs is a list of dics, e.g [{'buying': 'vhigh', 'maint': 'vhigh', 'doors': '2', 'persons': '2', 'lug_boot': 'small', 'safety': 'low'}, ...]
# ratings is a list of strins, e.g. ['unacc', 'unacc', 'good', ...]
# attributes is a list of all attributes of the car, except class, as this is the rating

# --- Random shuffle of data ---
data = list(zip(descs, ratings)) # pair descs with thier rating for random shuffle

random.seed(67) # set shuffle seed for reproducibility
random.shuffle(data)

# unzip back to desc and rating
descs, ratings = zip(*data) # turns into tuples
descs, ratings = list(descs), list(ratings) # converts back to lists

# --- Split data into train and test sets
splitRatio = 0.8 # 80% of the data is training and 20% is testing
change = int(splitRatio * len(ratings)) 
train_d, train_r, test_d, test_r = [],[],[],[]
for i in range(len(ratings)):
    if i < change:
        train_d.append(descs[i])
        train_r.append(ratings[i])
    else:
        test_d.append(descs[i])
        test_r.append(ratings[i])

# --- rating counts
# returns count of each rating e.g. {'unacc': 3, 'acc': 2, 'good': 1}
def count_ratings(ratings):
    counts = {} 
    for rating in ratings:
        if rating not in counts:
            counts[rating] = 0
        counts[rating] += 1
    return counts

# --- calculates entropy of ratings - lower = higher confidence
def entropy(ratings):
    counts = count_ratings(ratings)
    total = len(ratings) 
    ent = 0.0
    for rating, count in counts.items():
        p = count / total 
        calc = -p * math.log(p, 2)
        ent += calc
    return ent

# --- Partition attribute
def partition(descs, ratings, attribute):
    partitions = {} # initialise an empty dict

    for desc, rating in zip(descs, ratings): # loop through each row in dataset
        value = desc[attribute] # e.g. "low", "med", "high"
    
        # if value not seen, make new entry
        if value not in partitions:
            partitions[value] = ([],[]) # start with 2 empty lists
    
        # Add current row to right place
        desc_list, rating_list = partitions[value]
        desc_list.append(desc)
        rating_list.append(rating)
    
    return partitions
# Returns a dict: {attribute value: (list of desc with that attribute, list of ratings)}

# --- Weighted Entropy
# Returns the weighted entropy of an attribute to determine its information gain
def weighted_entropy(descs, ratings, attribute):
    if not descs or not ratings:
        return 0.0
    result = 0.0
    for value, (desc_list, rating_list) in partition(descs, ratings, attribute).items():
        value_size = len(desc_list) # number of instances where attribute label exists
        value_entropy = entropy(rating_list) # count of each rating
        weight = value_size / len(descs) # weight of value is value frequency / total dataset
        result += weight * value_entropy
    return result

# Returns the information gain of an attribute    
def information_gain(descs, ratings, attribute):
    return entropy(ratings) - weighted_entropy(descs, ratings, attribute)


# Tree data structure
class Node:
    def __init__(self):
        self.attribute = None # feature name used to split at this node
        self.children = {} # mapping from attribute value --> child node
        self.prediction = None # the class label if this node is a leaf
        self.majority_at_node = None # most common class among the rows that reach this node
        ## A leaf has prediction set and attribute unset, vice versa for an internal node

    def is_leaf(self): # Returns a boolean value whether it is a leaf node
        return self.prediction is not None
    
    def add_child(self, value, node): # adds a child node to the current node as a dict entry
        self.children[value] = node

    def get_child(self, value): # retrieves a child node with "value"
        return self.children.get(value)
    
    def set_majority(self, label): # sets the majority at node variable to a label of an attribute
        self.majority_at_node = label

# check if all values in ratings are identical
def all_identical(ratings):
    if not ratings:
        return False
    for label in ratings:
        if label != ratings[0]:
            return False
    return True

# returns majority label on a subset of ratings
def find_majority_label(ratings):
    counted_ratings = count_ratings(ratings)
    majority_label = max(counted_ratings, key=counted_ratings.get)
    return majority_label

# Global majority label used as default
global global_majority_rating
global_majority_rating = find_majority_label(train_r)

def build_tree(descs, ratings, attributes):
    # confim inputs align
    if not len(descs) == len(ratings):
        raise IndexError("Descs and ratings not the same")
    
    node = Node()
    # base case A, if descs is empty, return a leaf node with prediction as a sensible default
    if not descs:
        node.prediction = global_majority_rating
        return node
    
    # base case B, check if all values in ratings are identical, and return that rating if they are
    if all_identical(ratings):
        node.prediction = ratings[0]
        return node
    
    # base case C, check if attributes is empty, if so, return majority label in ratings
    if not attributes:
        node.prediction = find_majority_label(ratings)
        return node
    
    # find best attribute
    best_attr = None
    best_ig = 0
    for a in attributes:
        ig = information_gain(descs, ratings, a)
        if ig >= best_ig:
            best_ig = ig
            best_attr = a
    ### TO DO maybe add a rule where if best_ig or the subset is too small, return leaf node with prediction = majority label
    
    # if best_attr is still none, then every ig is exactly 0, thus set it as the majority label 
    if best_attr is None or best_ig <= 0:
        node.prediction = find_majority_label(ratings)
        return node

    # set node attribute to best attribute and set majority at node
    node.attribute = best_attr
    node.majority_at_node = find_majority_label(ratings)

    # partition data by best attribute
    parts = partition(descs, ratings, best_attr)
    for value, (desc_list, rating_list) in parts.items():
        # if ratings list is empty, create a leaf node with prediction as the majority rating at that attribute and attach to tree as a child of that attribute
        if not rating_list:
            child_node = Node()
            child_node.prediction = node.majority_at_node
            node.add_child(value, child_node)
            continue
        # if ratings are all the same create a child leaf node with the prediction as that rating
        if all_identical(rating_list):
            child_node = Node()
            child_node.prediction = rating_list[0]
            node.add_child(value, child_node)
            continue
        # Otherwise, split futher with all attributes execpt the best attribute
        remaining_attributes = []
        for attr in attributes:
            if attr != best_attr:
                remaining_attributes.append(attr)
        child_node = build_tree(desc_list, rating_list, remaining_attributes)
        node.add_child(value, child_node)
    return node

# Predict the rating of a desc, by walking the tree that was built
# root is the root node of the tree, desc is a description of a car, returns a rating
def predict_one(root: Node, desc: dict) -> str:
    node = root
    # return the prediction if we are at the leaf node, else move down tree
    while not node.is_leaf():
        # retrive attribute of node and find this attribute in desc
        attr = node.attribute

        # fallback if no attribute on non leaf node for some reason
        if attr is None:
            return node.majority_at_node
        
        value = desc.get(attr) # get value of attribute in desc
        child_node = node.get_child(value) # get child node
        if child_node is not None: # check if this value is in child branch
            # follow tree iteratively
            node = child_node
            continue
        else:
            # safety if value not in child branch
            return node.majority_at_node
    return node.prediction

# runs predictions for each desc and returns a list of predictions, can be mapped with descs to determine results and accuracy    
def predict_many(root: Node, descs: list[dict]) -> list:
    predictions = []
    for desc in descs:
        pred = predict_one(root, desc)
        predictions.append(pred)
    return predictions

### Testing

## Print Training and Test dataset sizes
print(f"Training set size: {len(train_d)}")
print(f"Test set size: {len(test_d)}")

## compute and print total accuracy
def calcAccuracy(predicted: list, actual: list):
    # check lists are same length
    if len(predicted) != len(actual):
        raise IndexError("predicted and actual are not same length")
    
    accuracy = 0.0
    correct_increase = 1/len(predicted)
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            accuracy += correct_increase
    return accuracy

train_tree = build_tree(train_d, train_r, attributes)
test_results = predict_many(train_tree, test_d)
total_accuracy = calcAccuracy(test_results, test_r)
print(f"Total acccuracy: {total_accuracy:.3f}")

## Per class metrics
# all metrics is a dict: {class: [tp, fp, tn, fn, precision, recall, f1_score], ...}
all_metrics = {}
for c in classes:
    # initialise metrics for the class
    tp, fp, tn, fn = 0,0,0,0
    for i in range(len(test_results)):
        pred = test_results[i]
        truth = test_r[i]
        # true positive
        if pred == c and truth == c:
            tp += 1
        # false positive
        elif pred == c and truth != c:
            fp += 1
        # false negative
        elif pred != c and truth == c:
            fn += 1
        # true negative
        else:
            tn += 1
    # precision is how many predicted positives are actually correct, high = few false alarms
    precision = tp / (tp + fp)
    # recall is how many real positives are caught, hig = few missed cases
    recall = tp / (tp + fn)
    # F1-score rewards balance between precision and recall
    f1_score = (2 * precision * recall) / (precision + recall)

    class_metric = [tp,fp,tn,fn, precision, recall, f1_score]
    all_metrics[c] = class_metric

# Macro averages for precision, recall and f1 score
macro_P = sum(all_metrics[c][4] for c in classes) / len(classes)
macro_R = sum(all_metrics[c][5] for c in classes) / len(classes)
macro_F = sum(all_metrics[c][6] for c in classes) / len(classes)

# Weighted averages for precision, recall and f1 score
def class_weight(c: list):
    return c[0] + c[3]
weighted_P = sum(all_metrics[c][4] * class_weight(all_metrics[c]) for c in classes) / sum(class_weight(all_metrics[c]) for c in classes)
weighted_R = sum(all_metrics[c][5] * class_weight(all_metrics[c]) for c in classes) / sum(class_weight(all_metrics[c]) for c in classes)
weighted_F = sum(all_metrics[c][6] * class_weight(all_metrics[c]) for c in classes) / sum(class_weight(all_metrics[c]) for c in classes)

# Print frequency of each class
print("Train set class counts:", {c: train_r.count(c) for c in classes})
print("Test set class counts:", {c: test_r.count(c) for c in classes})

## Print metrics
for c in classes:
    print(f"Class: {c}, Precision: {all_metrics[c][4]:.3f}, Recall: {all_metrics[c][5]:.3f}, F1-score: {all_metrics[c][6]:.3f}")
print(f"Macro Precision: {macro_P:.3f}, Macro Recall: {macro_R:.3f}, Macro F1-Score: {macro_F:.3f}")
print(f"Weighted Precision: {weighted_P:.3f}, Weighted Recall: {weighted_R:.3f}, Weighted F1-Score: {weighted_F:.3f}")

## Plot learning Curve

# create x axis in sets of 10%:
x = [] # % of training seed
for i in range(1,11):
    x.append(i*10) # % of training seed

x_numOfTrainingSets = []
y = [] # accuracy 0-1
for percentage in x:
    train_subset_d = train_d[0:int(len(train_d) * (percentage/100))]
    train_subset_r = train_r[0:int(len(train_r) * (percentage/100))]
    train_tree = build_tree(train_subset_d, train_subset_r, attributes)
    test_results = predict_many(train_tree, test_d)
    total_accuracy = calcAccuracy(test_results, test_r)
    y.append(total_accuracy)
    x_numOfTrainingSets.append(len(train_subset_d))

y_percentage = [a*100 for a in y] # convert y to a percentage

plt.plot(x_numOfTrainingSets, y_percentage, marker='o') # line with circular markers
plt.ylim(60, 100) # y axis to go from 0 to 100
plt.xlabel("Number of values in training set")
plt.ylabel("Accuracy (%)")
plt.title("Decision Tree Learning Curve")
plt.grid(True) # adds a grid
plt.show() # display window

