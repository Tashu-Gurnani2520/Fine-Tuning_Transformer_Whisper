# Whisper Fine-Tuning for Automatic Speech Recognition (ASR)

**Whisper** is an open-source, pre-trained Transformer-based encoder-decoder model developed for automatic speech recognition (ASR) and speech translation. This project focuses on fine-tuning Whisper for Hindi language transcription and experimenting with its capabilities for English transcription.


## Summary of Results

This project demonstrates the significant improvement achieved by fine-tuning the Whisper model:

| **Model**              | **WER (Word Error Rate)** | **Dataset**              | **Notes**                                          |
|------------------------|---------------------------|--------------------------|--------------------------------------------------|                           |
| Whisper (Small)        | 34%                      | Common Voice             | Fine-tuned for 1000 steps                        |
| Whisper (Small)        | 32%                      | Common Voice             | Fine-tuned for 2000 steps                        |
| Whisper (Medium)       | 20%                      | Google Fleurs Dataset    | Fine-tuned for 1000 steps                        |
| Whisper (English)      | 1%                       | Republic TV Audio        | Performed exceptionally without fine-tuning      |

## About Whisper

Whisper is an open-source, sequence-to-sequence model based on the **Transformer architecture**. It was pre-trained on **680k hours of labeled speech data** using large-scale weak supervision, enabling robust ASR performance across multiple languages.

The model is available in different sizes (**tiny**, **small**, **medium**, **large**), balancing performance and computational requirements. This project utilized the **small** and **medium** models, fine-tuning them for Hindi speech recognition.

## Fine-Tuning Process

### Dataset Selection
- **Common Voice Dataset**: Used for initial fine-tuning (WER reduced to 32%).
- **Google Fleurs Dataset**: A more robust dataset for Hindi, reducing WER to **20%** after 1000 steps.

### Hyperparameter Tuning

Fine-tuning was conducted using the **Transformers** library. The hyperparameters were optimized for training on a **GPU P100** provided by Kaggle.

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-hindi",  
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  
    learning_rate=1e-5,  
    warmup_steps=250,  
    max_steps=1000,  
    gradient_checkpointing=True,  
    fp16=True,  
    evaluation_strategy="steps",  
    per_device_eval_batch_size=4,  
    predict_with_generate=True,  
    generation_max_length=225,  
    save_steps=1000,  
    eval_steps=250,  
    logging_steps=25,  
    report_to=["tensorboard"],  
    load_best_model_at_end=True,  
    metric_for_best_model="wer",  
    greater_is_better=False,  
    push_to_hub=True,  
)

#Key Hyperparameters

##Learning Rate (learning_rate):
Determines how quickly the model updates weights during training. A smaller value (e.g., 1e-5) prevents overshooting the optimal solution, ensuring stable convergence.

##Warmup Steps (warmup_steps):
Gradually increases the learning rate from 0 to the defined value during these steps. Helps the model stabilize in the initial stages of training.

##Max Steps (max_steps):
The total number of training steps. For large datasets or complex models, more steps (e.g., 2000) are needed for convergence.

##Batch Size and Gradient Accumulation:
Determines how many samples are processed in one pass. Larger batch sizes require more GPU memory.
Gradient accumulation allows simulating larger batch sizes by accumulating gradients over multiple smaller batches.

Experimenting with these hyperparameters played a crucial role in achieving optimal performance. Specifically, the learning rate and warmup steps were adjusted iteratively to balance convergence speed and stability.

#English Model Performance
The Whisper English model, without fine-tuning, performed exceptionally on Republic TV audio, achieving 99% accuracy. Its ability to insert punctuation enhances the readability of transcripts, making it suitable for real-world applications.

#Usage

##Dependencies
Install the required libraries:

```python
pip install transformers datasets torchaudio

## Code for Transcription

Here’s how to use the fine-tuned model:

```python
from transformers import pipeline

# Load the model
pipe = pipeline(model="Tashuu/whisper-medium-hindi")

# Transcribe audio
def transcribe(file_path):
    text = pipe(file_path)["text"]
    return text

audio_file = "path_to_audio_file.wav"
print("Transcription:", transcribe(audio_file))

#References

Hugging Face Transformers
Whisper Model
Google Fleurs Dataset