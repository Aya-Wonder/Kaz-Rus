---
license: mit
datasets:
- issai/Kazakh_Speech_Corpus_2
language:
- kk
- ru
base_model: abilmansplus/whisper-turbo-ksc2
pipeline_tag: automatic-speech-recognition
library_name: peft
tags:
- base_model:adapter:abilmansplus/whisper-turbo-ksc2
- lora
- transformers
---

# Model Card for Model ID
This is a [**whisper-large-v3-turbo**](https://huggingface.co/openai/whisper-large-v3-turbo) speech-to-text model 
fine-tuned on [**Kazakh Speech Corpus 2 (ISSAI)**](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2) and [**Golos**](https://www.openslr.org/114/) datasets.  
It achieves great performance in **Kazakh** and decent performance in **Russian**.  

#### Word-Error-Rates (WERs) on test-sets
| Model                                 | ISSAI-KSC2  | CV-kaz | FLEURS-kaz | Golos-crowd  | CV-rus | FLEURS-rus |  
|-|-|-|-|-|-|-|
| abilmansplus/whisper-turbo-kaz-rus-v1 | **8.92%**  | **13.34%** | **13.60%**     | **8.95%**  | 20.73%     | 16.34%     |  
| openai/whisper-large-v3-turbo         | 70.30%     | 47.42%     | 23.68%         | 26.25%     | **8.78%**  |  **5.21%** |  

#### Character-Error-Rates (CERs) on test-sets
| Model                                 | ISSAI-KSC2  | CV-kaz | FLEURS-kaz | Golos-crowd  | CV-rus | FLEURS-rus |  
|-|-|-|-|-|-|-|
| abilmansplus/whisper-turbo-kaz-rus-v1 | **2.99%**  | **3.50%** | **5.43%**     | **2.42%**  | 5.52%     | 6.66%     |  
| openai/whisper-large-v3-turbo         | 34.26%     | 27.83%     | 6.02%         | 16.54%     | **3.61%**  |  **1.46%** |  

## Model Details
### Recommendations
Best suited for relatively clean Kazakh and Russian speech transcription.  
You may need to further fine-tune the model on your domain-specific datasets (e.g., phone-calls).  
The model outputs transcripts that do NOT include punctuation, capitalization, or time-stamps.  

## How to Get Started with the Model
**For longer audio** (35+ seconds), you can divide them into 30-second chunks, transcribe each chunk separately, and then merge the results.  
For better quality long-form transcription consider dividing audio into voiced segments using VAD solutions, for example:  
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)

Example implementation of a transcriber that can handle both short and long audio files:  
```python
## !!! INSTALL REQUIRED PACKAGES !!!
## "librosa>=0.11.0", "peft>=0.18.0", "torch>=2.9.1", "transformers>=4.57.1"
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class Transcriber:
    def __init__(
            self, 
            model_path="abilmansplus/whisper-turbo-kaz-rus-v1", 
            processor_path="openai/whisper-large-v3-turbo",  # converts audio into mel-spectrogram features
            device="cuda", 
            sampling_rate=16_000, 
            num_beams=5,
            chunk_length_s=30, stride_length_s=1,
            half_precision=True
        ):
        self.processor = WhisperProcessor.from_pretrained(
            processor_path,
            language=None, 
            task="transcribe"
        )
        self.half_precision = half_precision
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype = torch.float16 if half_precision else torch.float32
        )
        self.model = self.model.to(device)
        self.sr = sampling_rate
        self.num_beams=num_beams
        self.chunk_length_s = chunk_length_s  # chunk length in seconds
        self.stride_length_s = stride_length_s  # overlap between chunks in seconds
    
    def transcribe(self, audio_path: str) -> str:
        speech_array, sampling_rate = librosa.load(audio_path, sr=self.sr)
        audio_length_s = len(speech_array) / self.sr
        
        # If audio is shorter than chunk_length_s, process normally
        if audio_length_s <= self.chunk_length_s:
            full_transcription = self._transcribe_chunk(speech_array)
            return full_transcription
        
        # For longer audio, process in chunks
        chunk_length_samples = int(self.chunk_length_s * self.sr)
        stride_length_samples = int(self.stride_length_s * self.sr)

        # Calculate number of chunks
        num_samples = len(speech_array)
        num_chunks = max(1, 
                         int(
                             1 +
                             np.ceil(
                                     (num_samples - chunk_length_samples) / 
                                     (chunk_length_samples - stride_length_samples)
                                    ) 
                            )
                        )

        transcriptions = []

        for i in range(num_chunks):
            # Calculate chunk start and end
            start = max(0, i * (chunk_length_samples - stride_length_samples))
            end = min(num_samples, start + chunk_length_samples)
            
            # Get audio chunk
            chunk = speech_array[start:end]
            
            # Transcribe chunk
            chunk_transcription = self._transcribe_chunk(chunk)
            transcriptions.append(chunk_transcription)
        
        # Combine transcriptions (simple concatenation for now)
        full_transcription = " ".join(transcriptions)
        
        return full_transcription

    def _transcribe_chunk(self, audio_chunk) -> str:
        # Process inputs
        inputs = self.processor(
            audio_chunk, 
            sampling_rate=self.sr, 
            return_tensors="pt"
        ).input_features.to(self.model.device)

        if self.half_precision:
            inputs = inputs.half()
        
        # Get forced decoder IDs for language and task
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=None, 
            task="transcribe"
        )

        # The attention mask should be 1 for all positions in the input features
        attention_mask = torch.ones_like(inputs[:, :, 0])
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs, 
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=self.num_beams,
                attention_mask=attention_mask,
            )
        
        # Decode the generated IDs to text
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
```

## Training Details
### Training Data
Fine-tuning has been performed on the following datasets:  
- [**Kazakh Speech Corpus 2 (ISSAI)**](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2) - train split
- [**Golos**](https://www.openslr.org/114/) - "crowd" subset, train split
  - "farfield" subset was omitted because, depending on use-case, we may not want to transcribe far-away voices (e.g., background chatter).  

### Training Hyperparameters

```python
  model_path = "abilmansplus/whisper-turbo-ksc2"  # model to fine-tune
  processor_path = "openai/whisper-large-v3-turbo"  # this can stay constant, processor never changes, converts audio into mel-spectrogram
  train_batch_size = 4
  grad_accum_steps = 2
  learning_rate = 1e-5
  warmup_steps = 20000
  weight_decay = 0.01
  fp16 = True
  bf16 = False
  num_epochs = 3
  dataloader_num_workers = 8
  dataloader_prefetch_factor = 4
  sample_rate = 16000  # for audio
  max_audio_len = 30  # Maximum audio length in seconds
  gradient_checkpointing = True
  freeze_encoder = False  # if True, only Decoder is fine-tuned, saves a lot of time and GPU memory on less powerful machines
  ## LoRA params
  use_lora = True  # LoRA tecnique to reduce the number of trainable parameteres
  lora_r = 64  # rank
  lora_alpha = lora_r * 2
  lora_dropout = 0.05
```

## Evaluation

#### Testing Data

Testing has been done on the following datasets:  
- [**Kazakh Speech Corpus 2 (ISSAI)**](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2) - test split
- [**Golos**](https://www.openslr.org/114/) - "crowd" subset, ~5000 samples from test split (the rest were left-out for validation purposes)  
  - "farfield" subset was omitted because, depending on use-case, we may not want to transcribe far-away voices (e.g., background chatter).
- [**Common Voice (CV)**](https://commonvoice.mozilla.org/en) - Common Voice Scripted Speech 23.0
  - [Kazakh subset (CV-kaz)](https://datacollective.mozillafoundation.org/datasets/cmj8u3pbb00dhnxxbsqe4vbpc) - test split  
  - [Russian subset (CV-rus)](https://datacollective.mozillafoundation.org/datasets/cmj8u3prj00o9nxxbg5pbn88l) - test split
- [**FLEURS**](https://huggingface.co/datasets/google/fleurs) - Kazakh and Russian subsets, test splits  


#### Hardware
Single GPU: Nvidia RTX 5060 Ti 16GB  

### Framework versions

- PEFT 0.18.0