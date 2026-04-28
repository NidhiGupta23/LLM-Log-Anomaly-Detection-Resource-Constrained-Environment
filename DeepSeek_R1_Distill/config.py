from dataclasses import dataclass


@dataclass
class Config:
    # Model
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Dataset paths
    normal_file: str = "dataset/normal.log"
    abnormal_file: str = "dataset/abnormal.log"

    # Split
    test_size: float = 0.2
    random_seed: int = 42

    # Inference
    batch_size: int = 16
    max_length: int = 1024
    max_new_tokens: int = 3

    # Outputs
    output_misclassified: str = "misclassified.txt"
    output_confusion_matrix: str = "confusion_matrix.npy"
    output_metrics: str = "metrics.json"

