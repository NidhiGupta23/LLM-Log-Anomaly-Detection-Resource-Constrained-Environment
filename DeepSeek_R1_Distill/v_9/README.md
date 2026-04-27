## Results

EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.8753
  Precision : 0.3671
  Recall    : 1.0000
  F1-score  : 0.5370
  ROC-AUC   : 0.9328

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         322           50
  Actual Abnormal         0           29

Timing:
  Overall time   : 185.00 sec
  Avg per log    : 461.34 ms

Memory:
  Start RSS      : 7703.5 MB
  End RSS        : 7867.0 MB
  Peak RSS       : 7867.0 MB
  Delta RSS      : 163.5 MB
  System RAM     : 9.4%

Routing:
  Rule-based     : 276
  DeepSeek       : 125
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.87      0.93       372
Abnormal (1)       0.37      1.00      0.54        29

    accuracy                           0.88       401
   macro avg       0.68      0.93      0.73       401
weighted avg       0.95      0.88      0.90       401


Misclassified: 50 / 401
