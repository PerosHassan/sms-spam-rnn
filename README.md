# RNN SMS Spam Detection Classifier — Discovery-to-Action Strategy

## Project Overview
This project builds a sequence-aware Recurrent Neural Network (RNN) using TensorFlow/Keras to classify SMS messages as either legitimate communication (Ham) or unsolicited messaging (Spam). Incorporating dense word embeddings, regularized LSTM recurrent layers, and early termination strategies, the pipeline processes unstructured messaging strings to deliver highly precise classification routing.

---

## Technical Pipeline & Implementation Strategy

### 1. Discovery Phase (Data Preprocessing)
* **Target Harmonization:** Textual source tags are mapped into clean binary targets: `spam` $\rightarrow$ `1`, `ham` $\rightarrow$ `0`.
* **Tokenization:** Converts words into integer sequences using a Keras `Tokenizer` limited to the top 5,000 active words.
* **Sequence Standardization:** Pads input structures using `pad_sequences` to ensure a uniform length of 50 tokens, handling variable sentence structures efficiently.

### 2. Recurrent Deep Learning Architecture
* **Embedding Layer:** Maps token indexes into a continuous 32-dimensional semantic coordinate space.
* **LSTM Memory Gates:** Incorporates an `LSTM` layer to track contextual word relationships across sequences and eliminate vanishing gradient drops.
* **Overfitting Protection:** Integrates a `Dropout(0.2)` layer to prevent memorization of localized vocabulary noise.
* **Convergence Control:** Employs an `EarlyStopping` callback to monitor validation loss and automatically halt training at peak performance.

---

## Strategic Operational & UX Summary

### Why Precision Dictates Spam Filters
* **False Positives (Critical Failure):** Moving a genuine message (e.g., medical confirmation or family text) to the junk folder causes an immediate loss of important information.
* **False Negatives (Acceptable Overhead):** Letting a spam message slip into the main inbox is a minor annoyance that can be quickly skipped by the user.
* **UX Recommendation:** To protect the user experience, the system prioritizes **Precision** over accuracy. Setting a strict 95% confidence threshold ensures that legitimate messages are safely delivered to the main inbox.

### Out-of-Sample Triage Case Tests
The model evaluates three distinct text cases, accurately mapping conversational messages close to `0.0` while routing suspicious texts straight toward `1.0`:
1. *"Hey, are we still meeting for lunch?"* $\rightarrow$ **HAM (0)**
2. *"URGENT! Your account is locked. Click here to verify."* $\rightarrow$ **SPAM (1)**
3. *"Congratulations, you won a $500 gift card!"* $\rightarrow$ **SPAM (1)**
