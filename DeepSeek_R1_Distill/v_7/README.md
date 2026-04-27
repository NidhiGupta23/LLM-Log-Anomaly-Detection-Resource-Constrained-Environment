## Results

EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.8678
  Precision : 0.3333
  Recall    : 0.8276
  F1-score  : 0.4752
  ROC-AUC   : 0.8493

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         324           48
  Actual Abnormal         5           24

Timing:
  Overall time   : 213.62 sec
  Avg per log    : 532.72 ms

Memory:
  Start RSS      : 7703.7 MB
  End RSS        : 7881.1 MB
  Peak RSS       : 7881.1 MB
  Delta RSS      : 177.4 MB
  System RAM     : 9.4%

Routing:
  Rule-based     : 254
  DeepSeek       : 147
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       0.98      0.87      0.92       372
Abnormal (1)       0.33      0.83      0.48        29

    accuracy                           0.87       401
   macro avg       0.66      0.85      0.70       401
weighted avg       0.94      0.87      0.89       401


Misclassified: 53 / 401
