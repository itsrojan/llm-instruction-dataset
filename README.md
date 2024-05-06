# Project Overview

This project is about improving how language models understand and follow instructions. A basic dataset is changed by adding clear instructions to it, and this updated dataset is used to fine-tune a language model. The goal is to see if the model works better when fine-tuned with these new instructions.

## Description

This project explores the generation of instruction-based datasets for training natural language processing models, followed by model fine-tuning using these datasets. 
The primary objective is to convert a non-instruction-based dataset into one that includes clear, task-specific instructions, and then fine-tune a pre-trained language model on this new dataset. By comparing the performance of the fine-tuned model against the original model, this study aims to assess the impact of instruction-based training on model effectiveness. Results, analysis, and a comprehensive guide to replicating this study are detailed within this repository.

## Setup

Ensure Python 3.11 is installed before starting. Then, install the required libraries using the following command:

```bash
pip install datasets transformers peft trl tqdm
```

## Implementation Details

1. **Load the Model and Dataset:**
   The original pre-trained model:
   - [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
     
   The fine-tuned models on: 
    - instruction-based dataset. (Mistral-sentiment-fine-tuned)
    - combined dataset. (Mistral-mixed-fine-tuned)

2. **Dataset**
   - [Twitter Sentiment Analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
   - [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca?row=0)
     
3. **Run the Notebook:**
   - Open the notebook in Jupyter or another compatible environment and execute the cells sequentially.

4. **Fine-tune the Model:**
   - Add instructions to the twitter sentiment analysis dataset.
   - Follow the instructions in the notebook to fine-tune the model on both the Twitter Sentiment Analysis dataset and the Alpaca datasets.
   - After initial fine-tuning, the model will be saved locally for further use or evaluation.

6. **Evaluate the Model:**
   - Evaluate the models by comparing their accuracy, F1 score, precision, and recall to understand their performance across different training configurations.

## Results

| Model                                      | Accuracy | F1 Score  | Precision | Recall   |
|--------------------------------------------|----------|-----------|-----------|----------|
|                  sentiment_fine_tuned      | 0.92     | 0.913     | 0.913     | 0.913    |
|                  mixed_fine_tuned          | 0.88     | 0.864     | 0.905     | 0.826    |
|                  original_pretrained       | 0.62     | 0.689     | 0.553     | 0.913    |

- Both fine-tuned models significantly outperform the Original Pretrained Model in accuracy and precision, which shows the effectiveness of fine-tuning on specialized tasks.
- The Fine-Tuned Sentiment Model slightly outperforms the Mixed Fine-Tuned Model in overall effectiveness, likely due to its specialized focus. Meanwhile, the Mixed Model offers a balanced approach, handling a broad range of instructions effectively.

### 10 out-of-sample Instructions

The Sentiment Fine-Tuned Model typically gives accurate answers that closely match the input, demonstrating strong understanding but sometimes being too literal. In contrast, the Mixed Fine-Tuned Model offers more varied responses, adapting better to creative tasks like generating questions or writing poems. While both models outperform the original, the Sentiment model excels in tasks requiring detailed analysis of sentiments, whereas the Mixed model is more flexible, handling a broad range of instructions effectively. 
