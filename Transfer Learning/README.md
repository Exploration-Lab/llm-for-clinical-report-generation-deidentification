### De-Identification Task

De-identification is conceptually similar to a Named Entity Recognition (NER) task. We converted all the datasets into BIO format in the preprocessing step. The NER problem can be formulated as follows: given some text, \( S = (w_1, w_2, w_3, ..., w_n) \) containing \( n \) words, de-identification requires labeling each word \( w_i \) with a tag \( t_k \) from a NER tagset \( t_1, t_2, ..., t_T \). Subsequently, the labeled entities can be redacted or replaced with fake values for privacy protection.

### De-Identification Model

We fine-tuned several different NER models, including ghadeermobasher/BCHEM4-Modified-BioBERT-v1 (BioBERT) and Clinical-AI-Apollo/Medical-NER (Clinical AI Apollo), using a training partition of the data for training and a validation partition for evaluation. However, the Clinical NER models did not perform well as they are designed to label medical entities such as diseases, drugs, procedures, and devices. 

We utilized the RoBERTa-NER-Personal-Info model (PI-RoBERTa), which showed good performance on all the datasets. PI-RoBERTa is a 24-layer transformer model that predicts a label for each token. Architecture of PI-Roberta is give also provided.