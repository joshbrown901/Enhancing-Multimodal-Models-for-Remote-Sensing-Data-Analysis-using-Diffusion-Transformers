from sklearn.metrics import accuracy_score, f1_score, jaccard_score

def calculate_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    iou = jaccard




