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
