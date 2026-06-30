# V1 Model Evaluation Report on test2 Dataset

This report evaluates the performance of the **V1 (DINOv2 + Projection Head)** model on the **test2** paired dataset (100 Optical-SAR image pairs).

## Model Configuration
* **Core Backbone**: `facebook/dinov2-base` (768-dimensional outputs)
* **Projection Heads**: Two-layer PyTorch MLP projection heads mapping 768-D inputs into a shared 256-D space.
* **Weights loaded from**:
  * Optical projection: `backend\cache\opt_proj.pt`
  * SAR projection: `backend\cache\sar_proj.pt`

## Dataset Summary
* **Dataset**: `test2`
* **Size**: 100 paired Optical & SAR images.
* **Protocol**: 1-to-1 retrieval. For each Optical image, we retrieve its corresponding SAR image from the 100-image SAR gallery.

---

## 1. Retrieval Metrics

The table below summarizes the retrieval metrics calculated over the 100 queries against the 100 SAR gallery images.

| Metric | Value | Description |
|---|---|---|
| **Recall@1** | 0.3800 (38.0%) | Success rate of the correct match being at rank 1 |
| **Recall@5** | 0.7500 (75.0%) | Success rate of the correct match being in the top 5 |
| **Recall@10** | 0.8400 (84.0%) | Success rate of the correct match being in the top 10 |
| **Precision@1** | 0.3800 | Average precision at rank 1 |
| **Precision@5** | 0.1500 | Average precision at rank 5 |
| **Precision@10** | 0.0840 | Average precision at rank 10 |
| **F1-Score@1** | 0.3800 | Harmonic mean of Precision@1 and Recall@1 |
| **F1-Score@5** | 0.2500 | Harmonic mean of Precision@5 and Recall@5 |
| **F1-Score@10** | 0.1527 | Harmonic mean of Precision@10 and Recall@10 |
| **Mean Average Precision (mAP)** | 0.5318 | Average Precision over all queries |
| **Mean Reciprocal Rank (MRR)** | 0.5318 | Reciprocal rank of the first correct retrieval |
| **Top-1 Accuracy** | 38.00% | Same as Recall@1 (1-to-1 retrieval) |
| **Top-5 Accuracy** | 75.00% | Same as Recall@5 (1-to-1 retrieval) |
| **Top-10 Accuracy** | 84.00% | Same as Recall@10 (1-to-1 retrieval) |

*Note: In a 1-to-1 matching setup, Recall@K is mathematically identical to Top-K Accuracy, and mAP is mathematically identical to MRR, because there is exactly one correct match in the gallery for each query.*

---

## 2. Latency Analysis

*Measurements were performed on the hardware used for evaluation (Device: `CPU`).*

* **Average Retrieval Latency (End-to-End)**: **218.50 ms** per query.

### Sub-step Latency Breakdown

| Phase | Average Time (ms) | Percentage of Total |
|---|---|---|
| **Optical Preprocessing** | 4.42 ms | 2.0% |
| **DINOv2 Feature Extraction** | 213.57 ms | 97.7% |
| **Optical Projection Head** | 0.41 ms | 0.2% |
| **Similarity Search & Ranking** | 0.09 ms | 0.0% |
| **Total** | **218.50 ms** | **100.0%** |

---

## 3. Binary Classification Metrics (Pairwise Match/Non-match)

By treating each of the $100 \times 100 = 10,000$ possible Optical-SAR pairs as a binary classification decision (Match = 1, Non-match = 0), we evaluate model performance under two threshold strategies:

### Strategy A: Optimal Threshold (Maximizing F1-Score)
* **Optimal Threshold**: `0.5723`

| Metric | Value | Description |
|---|---|---|
| **Precision** | 0.4615 | True matches / predicted matches |
| **Recall** | 0.3600 | True matches / actual matches |
| **F1 Score** | 0.4045 | Harmonic mean of Precision and Recall |
| **Accuracy** | 0.9894 (98.94%) | Total correct predictions / total pairs |

#### Confusion Matrix
* **True Positives (TP)**: 36
* **False Positives (FP)**: 42
* **False Negatives (FN)**: 64
* **True Negatives (TN)**: 9858

```
                      Predicted Match    Predicted Non-Match
Actual Match             36              64                 
Actual Non-Match         42              9858               
```

### Strategy B: Default Threshold (0.50)
* **Threshold**: `0.5000`

| Metric | Value |
|---|---|
| **Precision** | 0.2717 |
| **Recall** | 0.5000 |
| **F1 Score** | 0.3521 |
| **Accuracy** | 0.9816 (98.16%) |

#### Confusion Matrix
* **True Positives (TP)**: 50
* **False Positives (FP)**: 134
* **False Negatives (FN)**: 50
* **True Negatives (TN)**: 9766

```
                      Predicted Match    Predicted Non-Match
Actual Match             50              50                 
Actual Non-Match         134             9766               
```

---

## 4. Key Observations & Insights
1. **Modality Matching Capability**: With the correct preprocessing (matching the V1 training pipeline), the model achieves a **Top-1 Accuracy of 38.0%**, **Top-5 Accuracy of 75.0%**, and a **Top-10 Accuracy of 84.0%** on the `test2` dataset. This is a massive improvement over the random chance baseline (1% Top-1) and indicates that the V1 model does possess a significant modality-alignment capacity.
2. **Backbone Bottleneck**: Latency is heavily dominated by the **DINOv2 backbone feature extraction**, which accounts for **97.7%** of the total retrieval time (213.57 ms out of 218.50 ms). In contrast, the Projection Head and the Similarity Search are extremely lightweight, requiring less than 1ms combined.
3. **Similarity Distribution and Thresholding**: The similarity scores are much better aligned when preprocessed correctly, centering between 0.3 and 0.6. At the default threshold of `0.5000`, the model achieves a pairwise F1-Score of `0.3521` (with 50 true positive matches and 134 false positives). Optimizing the threshold to `0.5723` reduces false positives to 42, which significantly improves the pairwise F1-Score to `0.4045`.
