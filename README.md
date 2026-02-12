# NeuroScript-LLM: Leveraging a Large Language Model Architecture for Automated Alzheimer's Disease Clinical Synthesis
## Dataset
We provide results on the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, The data is available on their website via this link [The ADNI dataset](http://adni.loni.usc.edu) 

## Description
In this work, we propose NeuroScript-LLM, a lightweight large language model architecture designed to automate the generation of comprehensive clinical reports directly from textual patient data. Our approach converts heterogeneous clinical information—including cognitive assessments, biomarker results, genetic profiles, and quantitative neuroimaging findings—into a unified textual representation that feeds into a fine-tuned T5-based generator. By relying exclusively on text inputs, NeuroScript-LLM eliminates the need for heavy multimodal encoders or raw 3D image processing, resulting in a parameter-efficient model that is computationally accessible and practical for real-world deployment. Experimental evaluation on the ADNI dataset demonstrates that NeuroScript-LLM generates reports closely aligned with expert references, achieving a BERTScore of 0.9120, a ROUGE-L F1 score of 0.4932, BLEU-1 to BLEU-4 scores ranging from 0.5761 to 0.1626, and a METEOR score of 0.4681—substantially outperforming baseline generation models across all metrics.
<img width="1022" height="541" alt="Image" src="https://github.com/user-attachments/assets/dc8f90b5-e501-4b9d-91f4-578dde2f3690" />
## Data preparation
Run the dataset_text_only_.py notebook to organize the dataset for training.
## T5_generator model
Run the t5_generator.py notebook to load the T5 model and prepare it for report generation.
## Training of the model 
Run the run_medblip_text_only_.py notebook to train the model.
## Evaluation of the model
Use the test_model_text.py notebook to test the model and compute performance metrics.
