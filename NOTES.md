# NOTES

1. Our model uses temporal context, running energy medians, zero crossing rate, spectral centroid, spectral roll-off, energy decay slopes, final-syllable lengthening, and 26 voice texture MFCCs.
2. We added Pitch Delta (final 200ms pitch slope) and Breath Detector (ZCR spike and energy drop in final 100ms) to capture local turn-yielding cues.
3. Feature extraction is shared between training and prediction (`features.py`), and a programmatic causality invariance test verified it is strictly causal.
4. The final model is an unscaled RandomForest ensemble configured with 100 decision trees and 1:10 false-cutoff penalization to handle both languages without scaling-warping.
5. Predict.py writes raw continuous probabilities `p_eot` to allow external calibration.
6. In honest 5-fold cross-validation, the model achieves an OOF AUC of 0.6719 and beats the starter model's honest English OOF delay (1137 ms vs. 1305 ms).
7. Hindi OOF Delay (850 ms) matches the silence baseline because the penalization forces the model to be extremely conservative to avoid false cutoffs.
8. The model still fails on EOT pauses that end with flat/rising intonations (questions or level tones) or when EOT pauses are unusually short (e.g., 300ms).
9. With one more day, we would implement sequence-based models (LSTMs/GRUs) to capture pause timing context across the entire turn.
