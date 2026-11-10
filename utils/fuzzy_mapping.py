# utils/fuzzy_mapping.py
"""
Fuzzy Logic & FEM Boundary Conditions for Labels.
Instead of dropping anomalies (the "foam"), we map them to soft probabilities.
This mimics FEM, where boundaries are not black-and-white but have gradual transitions.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_fuzzy_labels(row, num_classes_rec=3, num_classes_band=3):
    """
    Maps a single row's target into a probability distribution.
    Handling the "Bohlool" anomalies (rate=0 with recommended).
    """
    # Initial uniform distribution (Quantum Superposition)
    rec_probs = np.ones(num_classes_rec) / num_classes_rec
    band_probs = np.ones(num_classes_band) / num_classes_band
    
    # --- FEM Boundary Condition 1: Recommendation ---
    # 0: recommended, 1: not_recommended, 2: no_idea
    rec_label = row.get('recommendation_status')
    if pd.notna(rec_label):
        # If the label exists, give it high probability (Strong boundary)
        # But we leave a little room for ambiguity (Fuzzy margin)
        rec_probs = np.array([0.05, 0.05, 0.05]) # base noise
        rec_probs[{'recommended': 0, 'not_recommended': 1, 'no_idea': 2}[rec_label]] = 0.90

    # --- FEM Boundary Condition 2: Rating Band & Rate Anomaly ---
    rate = row.get('rate')
    if pd.notna(rate):
        if rate == 0:
            # ANOMALY (The "Bohlool" madness): rate=0
            # It means "missing", but it carries semantic density.
            # If the user said 'recommended', we fuzz it towards 'medium/high'.
            if rec_label == 'recommended':
                band_probs = np.array([0.1, 0.4, 0.5]) # Fuzzy: mostly high, some medium
            elif rec_label == 'not_recommended':
                band_probs = np.array([0.5, 0.4, 0.1]) # Fuzzy: mostly low
            else:
                band_probs = np.array([0.33, 0.34, 0.33]) # Completely uncertain
        else:
            # Normal rate: apply hard thresholds but soften the edges
            if rate < 2.5: band_probs = np.array([0.85, 0.10, 0.05])
            elif rate < 4.0: band_probs = np.array([0.10, 0.80, 0.10])
            else: band_probs = np.array([0.05, 0.10, 0.85])

    return torch.tensor(rec_probs, dtype=torch.float32), torch.tensor(band_probs, dtype=torch.float32)


class FuzzyCrossEntropyLoss(nn.Module):
    """
    A custom loss that accepts soft targets (probability distributions).
    It acts like the "Weak Form" in FEM, integrating over the elements.
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, soft_targets):
        """
        logits: (batch, num_classes)
        soft_targets: (batch, num_classes) - probabilities summing to 1
        """
        log_probs = self.log_softmax(logits)
        # Cross entropy with soft targets: - sum(target * log(prob))
        loss = -torch.sum(soft_targets * log_probs, dim=-1)
        return loss.mean()