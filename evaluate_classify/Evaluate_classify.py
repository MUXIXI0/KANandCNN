from sklearn.metrics import confusion_matrix, classification_report
def Confusion_matrixs(y_true, y_pred):
    y_pred = [y_pred_index.view(-1).tolist() for y_pred_index in y_pred]
    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [y_true_index.view(-1).tolist() for y_true_index in y_true]
    y_true = [item for sublist in y_true for item in sublist]
    print(confusion_matrix(y_true, y_pred))

# 召回率等二分类任务指标
def Confusion_metrics(y_true, y_pred):
    y_pred = [y_pred_index.view(-1).tolist() for y_pred_index in y_pred]
    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [y_true_index.view(-1).tolist() for y_true_index in y_true]
    y_true = [item for sublist in y_true for item in sublist]
    print(classification_report(y_true, y_pred, digits=5))
