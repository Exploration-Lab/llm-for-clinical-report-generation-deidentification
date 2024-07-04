# LLM for Clinical Report Generation-Deidentification
Install the depedency in requirment.txt 

!pip install *requirement.txt

We aim to de-identify medical texts from the Indian Medical Institute. We initially identify the The present study takes a step towards answering this imminent need. Using a dataset of fully de-identified 99 discharge summaries obtained under Institutional Review Board (IRB) approval from the Sanjay Gandhi Post Graduate Institute of Medical Sciences (SGPGIMS), Lucknow, India, the study evaluates language models (LMs) for the task of de-identification; further, commercially available de-identification solutions are evaluated. Hereafter, we refer to this dataset as the Indian Clinical Discharge Summaries (ICDSR, subscript R refers to real) dataset. Given the paucity of clinical data, the study also evaluates Large Language Models (LLMs) on the task of generating synthetic clinical texts for training de-identification models. Critically, the study highlights the existence of several personal identifiable information (PII) elements in the ICDSR dataset that are unique to the language use and cultural practices in India. It is unlikely that the existing de-identification solutions have been trained to recognize these unique PII elements, and therefore, their detection may be unreliable. In a nutshell, we make the following contributions:
• We introduce a new dataset (Indian Clinical Discharge Summaries (ICDSR)) obtained from an Indian hospital and evaluate the performance of PI-RoBERTa model (PI-RoBERTa) (fine-tuned on non-Indian clinical summaries) on ICDSR for the task of De-Identification. Our experiments show poor cross-institutional performance. Experiments with existing commercial off-the-shelf clinical de-identification systems show similar trends.
• To overcome the paucity of Indian clinical data, we generate synthetic summaries using LLMs (Gemini (Team et al., 2023), Gemma(Team et al., 2024), Mistral (Jiang et al., 2023), and Llama3 (Touvron et al., 2023)) via In-Context learning (ICL). Further, the synthetic summaries are used to train PI-RoBERTa for de-identification on ICDSR. Results show significant improvement in the performance of the de-identification system.
# Preprocessing
We have five datasets: n2c2 2006, n2c2 2014, ICDS-R (real dataset from SGPGI), ICDS-G(l) (Llama-generated discharge summaries), and ICDS-G(g) (Gemini-generated discharge summaries). The Preprocessing folder contains preprocessing scripts for all the datasets.

# PII Detection using commercial tools and LLMs

We used LLMs (Llama-3-8B, Gemma-1.1-7b-it, Mistral-7B-Instruct-v0.1) and commercial tools (AWS Comprehend Medical API, GCP's Inspect) to detect PII on ICDS-R and evaluated their performance.


# Analysis and Comparision of Datasets

The notebooks in the Analysis and Comparision of Dtaset folders were used to analyze various biomedical real and synthetic datasets against each other in terms of their n-gram frequencies, Jaccard distance, and BERTScore (using bio-bert). The notebooks also yielded visualizations for the n-gram frequencies. Depending on the type of file format (json, xml etc), they were parsed and stored in strings, and then iterated through for further lexical analysis.
 

# DataGeneration
We have generated two datasets using in-context learning. We utilized Gemini-pro-1.0 and Llama-3-8B for generating discharge summaries. We generated 1,596 discharge summaries from Gemini-pro-1.0 using the n2c2 2006 dataset with in-context learning, producing around five discharge summaries per original summary. From Llama-3-8B, we generated 1,043 discharge summaries using the ICDS-R dataset, producing around 20 discharge summaries per original summary. The codes for generating the discharge summaries are provided in the Data Generation folder.

# Inter Annotator Agreement



# Transfer learning

### De-Identification Task

De-identification is conceptually similar to a Named Entity Recognition (NER) task. We converted all the datasets into BIO format in the preprocessing step. The NER problem can be formulated as follows: given some text, \( S = (w_1, w_2, w_3, ..., w_n) \) containing \( n \) words, de-identification requires labeling each word \( w_i \) with a tag \( t_k \) from a NER tagset \( t_1, t_2, ..., t_T \). Subsequently, the labeled entities can be redacted or replaced with fake values for privacy protection.

### De-Identification Model

We fine-tuned several different NER models, including ghadeermobasher/BCHEM4-Modified-BioBERT-v1 (BioBERT) and Clinical-AI-Apollo/Medical-NER (Clinical AI Apollo), using a training partition of the data for training and a validation partition for evaluation. However, the Clinical NER models did not perform well as they are designed to label medical entities such as diseases, drugs, procedures, and devices. 

We utilized the RoBERTa-NER-Personal-Info model (PI-RoBERTa), which showed good performance on all the datasets. PI-RoBERTa is a 24-layer transformer model that predicts a label for each token.

