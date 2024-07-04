# LLM for Clinical Report Generation-Deidentification
Install the depedency in requirment.txt 

!pip install *requirement.txt

## Main Contribution
Our contributions can be summarized as below:

• We introduce a new dataset (Indian Clinical Discharge Summaries (ICDSR)) obtained from an Indian hospital and evaluate the performance of PI-RoBERTa model (PI-RoBERTa) (fine-tuned on non-Indian clinical summaries) on ICDSR for the task of De-Identification. Our experiments show poor cross-institutional performance. Experiments with existing commercial off-the-shelf clinical de-identification systems show similar trends.

• To overcome the paucity of Indian clinical data, we generate synthetic summaries using LLMs Gemini , Gemma, Mistral , and Llama3 via In-Context learning (ICL). Further, the synthetic summaries are used to train PI-RoBERTa for de-identification on ICDSR. Results show significant improvement in the performance of the de-identification system.

## Citation
