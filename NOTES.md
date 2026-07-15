# NOTES

1. Our model uses temporal context, running energy medians, zero crossing rate, spectral centroid, spectral roll-off, energy decay slopes, and final-syllable lengthening.
2. We added Pitch Delta (final 200ms pitch slope) and Breath Detector (ZCR spike and energy drop in final 100ms) to capture local turn-yielding cues.
3. Feature extraction is shared between training and prediction (`features.py`), and a programmatic causality invariance test verified it is strictly causal.
4. The final model is an RBF Support Vector Classifier which outputs raw continuous probabilities `p_eot` without internal thresholding to allow for external calibration.
5. In honest 5-fold cross-validation, the model achieves an OOF AUC of 0.6749 (std: 0.0376) and beats the starter model's honest English OOF delay (1163 ms vs. 1305 ms).
6. Side-by-side comparison of full vs. pruned features confirms that the full-feature model generalizes better (1163 ms vs. 1230 ms).
7. Hindi OOF Delay (850 ms) is lower than English because Hindi true/hold pauses are systematically shorter.
8. The model still fails on EOT pauses that end with flat/rising intonations (questions or level tones) or when EOT pauses are unusually short (e.g., 300ms).
9. With one more day, we would implement sequence-based models (LSTMs/GRUs) to capture pause timing context across the entire turn.
