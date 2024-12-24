
• A seq2seq encoder-decoder model (trained on the KP20k dataset) generates keyphrases from the
training samples, extracting diverse and informative targets, even those not explicitly annotated
in the original data.
• For each generated keyphrase, the corresponding text is augmented, creating synthetic training
examples. This process ensures exposTTS: A Target-based Teacher-Student Framework for Zero- Shot Stance Detection 
Introduction:

The Target-Based Teacher-Student (TTS) framework represents a cutting-edge solution for Zero-
Shot Stance Detection (ZSSD), designed to classify stances (Pro, Con, Neutral) for previously unseen
targets. This framework innovatively combines target augmentation, pseudo-labeling, and a teacher-
student learning paradigm to overcome the challenges of limited human annotations and real-world ap-
plicability. Additionally, it extends its capabilities to an open-world ZSSD setting, where no human-
annotated targets or stance labels are available.
Datasets:

The framework is benchmarked on the VAried Stance Targets (VAST) dataset, which includes di-
verse themes such as politics, education, and public health. The dataset ensures no overlap between
training and testing targets, making it ideal for evaluating ZSSD. The augmented training set intro-
duces 7,406 new samples, created using keyphrase generation to diversify target coverage and improve
generalization.
Methodology
Target-Based Data Augmentation:
ure to a broader set of target-text combinations, facilitating
better adaptation to unseen targets.
Teacher-Student Learning:

• Teacher Model: A BART encoder fine-tuned on the original dataset predicts pseudo-labels (Pro,
Con, Neutral) for the augmented samples.
• Student Model: Trained on both the original dataset and the pseudo-labeled augmented dataset,
the student model incorporates the enriched target-text relationships learned from the teacher,
enhancing its capacity to detect stances for new targets.Student also uses BART encoder
• Iterative refinement between teacher and student ensures label quality and mitigates noise intro-
duced during data augmentation.
1
Open-World ZSSD:

• The task is redefined as a Natural Language Inference (NLI) problem, where stances are
mapped to entailment categories: Pro to Entailment, Con to Contradiction, and Neutral to Neutral.
• The BART-MNLI encoder interprets the text as a ”premise” and the target-based prompt (e.g., ”I
am in favor of [target]!”) as a ”hypothesis,” leveraging its pre-trained NLI capabilities to identify
stance relationships without requiring human-annotated labels.
• Augmented data undergoes confidence filtering, retaining only high-quality pseudo-labels to prevent
error propagation.

Training Architecture and Configurations:

• The BART-large model serves as the backbone for both teacher and student roles. The BART
encoder is fine-tuned, while the decoder is excluded to optimize memory usage.
• The framework employs adaptive learning rates and utilizes early stopping with validation-based
patience, ensuring efficient training.
• Synthetic datasets, built through augmented and filtered samples, serve as the cornerstone for
open-world ZSSD, allowing stance detection models to generalize effectively.
