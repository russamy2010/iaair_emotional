"""
Emotional Intelligence LLM Backend
A neural network-based system for emotion-aware conversational AI with audio input support
Integrated with CREMA-D and JL-Corpus datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd
import numpy as np
import json
import uuid
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import sqlite3
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import asyncio
import aiofiles
import librosa
import soundfile as sf
import io
from pydub import AudioSegment
import tempfile
import os
import subprocess
import requests
import zipfile
from pathlib import Path
import glob
import cv2
from PIL import Image
from tqdm import tqdm

# Handle optional imports with fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI Whisper not available. Installing...")
    try:
        subprocess.check_call(["pip", "install", "openai-whisper"])
        import whisper
        WHISPER_AVAILABLE = True
        print("✅ Whisper installed successfully!")
    except Exception as e:
        print(f"❌ Could not install Whisper: {e}")
        WHISPER_AVAILABLE = False
        whisper = None

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    print("Warning: Kagglehub not available. Installing...")
    try:
        subprocess.check_call(["pip", "install", "kagglehub"])
        import kagglehub
        KAGGLEHUB_AVAILABLE = True
        print("✅ Kagglehub installed successfully!")
    except Exception as e:
        print(f"❌ Could not install Kagglehub: {e}")
        KAGGLEHUB_AVAILABLE = False
        kagglehub = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class ModelConfig:
    """Configuration for the emotional intelligence model"""
    base_model: str = "microsoft/DialoGPT-medium"
    audio_model: str = "facebook/wav2vec2-large-960h-lv60-self"
    whisper_model: str = "base"  # Options: tiny, base, small, medium, large
    emotion_classes: List[str] = None
    max_sequence_length: int = 512
    max_audio_length: int = 16000 * 30  # 30 seconds at 16kHz
    hidden_size: int = 1024  # DialoGPT-medium has 1024 size
    audio_hidden_size: int = 1024  # Wav2Vec2 hidden size
    num_attention_heads: int = 16
    num_hidden_layers: int = 6
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate: int = 16000

    # Gradient accumulation
    gradient_accumulation_steps: int = 4  # New parameter for gradient accumulation

    # Dataset paths
    crema_d_path: str = "./datasets/crema_d"
    jl_corpus_path: str = "./datasets/jl_corpus"

    def __post_init__(self):
        if self.emotion_classes is None:
            # Map to your existing emotion classes
            self.emotion_classes = [
                "Joy", "Sadness", "Acceptance", "Disgust",
                "Fear", "Anger", "Surprise", "Anticipation", "Neutral"
            ]

class DatasetDownloader:
    """Handle downloading and preprocessing of training datasets"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.crema_d_path = Path(config.crema_d_path)
        self.jl_corpus_path = Path(config.crema_d_path).parent / "jl_corpus"

        # Create dataset directories
        self.crema_d_path.mkdir(parents=True, exist_ok=True)
        self.jl_corpus_path.mkdir(parents=True, exist_ok=True)

        # Emotion mapping dictionaries
        self.crema_d_emotion_map = {
            'ANG': 'Anger',
            'DIS': 'Disgust',
            'FEA': 'Fear',
            'HAP': 'Joy',
            'NEU': 'Neutral',
            'SAD': 'Sadness'
        }

        # JL-Corpus emotion mapping (based on the dataset description)
        self.jl_corpus_emotion_map = {
            'angry': 'Anger',
            'disgusted': 'Disgust',
            'fearful': 'Fear',
            'happy': 'Joy',
            'neutral': 'Neutral',
            'sad': 'Sadness',
            'surprised': 'Surprise',
            'anger': 'Anger',
            'disgust': 'Disgust',
            'fear': 'Fear',
            'happiness': 'Joy',
            'sadness': 'Sadness',
            'surprise': 'Surprise'
        }

    def download_crema_d(self) -> bool:
        """Download CREMA-D dataset using kagglehub"""
        if not KAGGLEHUB_AVAILABLE or kagglehub is None:
            logger.warning("⚠️ Kagglehub not available - CREMA-D download skipped")
            return False

        try:
            logger.info("Downloading CREMA-D dataset...")

            # Download using kagglehub
            path = kagglehub.dataset_download("ejlok1/cremad")
            logger.info(f"CREMA-D dataset downloaded to: {path}")

            # Copy/move files to our designated path
            import shutil
            source_path = Path(path)

            # Find audio files and copy them
            audio_files = list(source_path.glob("**/*.wav"))
            logger.info(f"Found {len(audio_files)} audio files in CREMA-D dataset")

            for audio_file in audio_files:
                dest_file = self.crema_d_path / audio_file.name
                if not dest_file.exists():
                    shutil.copy2(audio_file, dest_file)

            logger.info(f"CREMA-D dataset setup complete: {len(audio_files)} files")
            return True

        except Exception as e:
            logger.error(f"Error downloading CREMA-D dataset: {str(e)}")
            return False

    def download_jl_corpus(self) -> bool:
        """Download JL-Corpus dataset using kagglehub"""
        if not KAGGLEHUB_AVAILABLE or kagglehub is None:
            logger.warning("⚠️ Kagglehub not available - JL-Corpus download skipped")
            return False

        try:
            logger.info("Downloading JL-Corpus dataset...")

            # Download using kagglehub
            path = kagglehub.dataset_download("tli725/jl-corpus")
            logger.info(f"JL-Corpus dataset downloaded to: {path}")

            # Copy/move files to our designated path
            import shutil
            source_path = Path(path)

            # Find all files and copy them
            all_files = list(source_path.glob("**/*"))
            audio_files = [f for f in all_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']]
            csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
            
            logger.info(f"Found {len(audio_files)} audio files and {len(csv_files)} CSV files in JL-Corpus dataset")

            # Copy audio files
            audio_dest = self.jl_corpus_path / "audio"
            audio_dest.mkdir(exist_ok=True)
            
            for audio_file in audio_files:
                dest_file = audio_dest / audio_file.name
                if not dest_file.exists():
                    shutil.copy2(audio_file, dest_file)

            # Copy CSV files (labels/metadata)
            for csv_file in csv_files:
                dest_file = self.jl_corpus_path / csv_file.name
                if not dest_file.exists():
                    shutil.copy2(csv_file, dest_file)

            logger.info(f"JL-Corpus dataset setup complete: {len(audio_files)} audio files, {len(csv_files)} CSV files")
            return True

        except Exception as e:
            logger.error(f"Error downloading JL-Corpus dataset: {str(e)}")
            return False

    def process_crema_d_data(self) -> List[Dict]:
        """Process CREMA-D dataset into training format"""
        try:
            audio_files = list(self.crema_d_path.glob("*.wav"))
            processed_data = []

            logger.info(f"Processing {len(audio_files)} CREMA-D audio files...")

            for audio_file in tqdm(audio_files, desc="Processing CREMA-D"):
                try:
                    # Parse filename: ActorID_SentenceID_Emotion_Intensity
                    filename_parts = audio_file.stem.split('_')
                    if len(filename_parts) >= 3:
                        emotion_code = filename_parts[2]
                        emotion = self.crema_d_emotion_map.get(emotion_code, 'Neutral')

                        # Load and process audio
                        audio, sr = librosa.load(audio_file, sr=self.config.sample_rate)

                        # Generate a simple context message for training
                        context_messages = [
                            "Please respond to this emotional expression.",
                            "How would you respond to someone feeling this way?",
                            "What would be an appropriate emotional response?",
                            "Please acknowledge this person's emotional state."
                        ]

                        processed_data.append({
                            'id': str(uuid.uuid4()),
                            'audio_file': str(audio_file),
                            'user_message': f"[Audio expression of {emotion.lower()}]",
                            'response': f"I can hear that you're feeling {emotion.lower()}. I understand and I'm here to help.",
                            'context': np.random.choice(context_messages),
                            'emotion': emotion,
                            'audio_features': audio,
                            'dataset_source': 'crema_d',
                            'timestamp': datetime.datetime.now().isoformat()
                        })

                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {str(e)}")
                    continue

            logger.info(f"Successfully processed {len(processed_data)} CREMA-D samples")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing CREMA-D data: {str(e)}")
            return []

    def process_jl_corpus_data(self) -> List[Dict]:
        """Process JL-Corpus dataset into training format"""
        try:
            audio_dir = self.jl_corpus_path / "audio"
            processed_data = []

            # Look for CSV files with labels/metadata
            csv_files = list(self.jl_corpus_path.glob("*.csv"))
            
            if csv_files:
                # Process using CSV metadata
                logger.info(f"Processing JL-Corpus with {len(csv_files)} CSV files...")
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        logger.info(f"CSV columns: {df.columns.tolist()}")
                        
                        for _, row in tqdm(df.iterrows(), desc=f"Processing {csv_file.name}", total=len(df)):
                            try:
                                # Try to extract emotion and filename from different possible column names
                                emotion = None
                                audio_filename = None
                                utterance = ""
                                
                                # Common column name variations for emotion
                                emotion_cols = ['emotion', 'label', 'sentiment', 'feeling', 'class']
                                for col in emotion_cols:
                                    if col in df.columns:
                                        emotion_raw = str(row[col]).lower().strip()
                                        emotion = self.jl_corpus_emotion_map.get(emotion_raw, emotion_raw.title())
                                        break
                                
                                # Common column name variations for filename
                                filename_cols = ['filename', 'file', 'audio_file', 'path', 'wav_file']
                                for col in filename_cols:
                                    if col in df.columns:
                                        audio_filename = str(row[col])
                                        break
                                
                                # Common column name variations for text/utterance
                                text_cols = ['text', 'utterance', 'transcript', 'sentence', 'content']
                                for col in text_cols:
                                    if col in df.columns and pd.notna(row[col]):
                                        utterance = str(row[col])
                                        break
                                
                                if not emotion:
                                    # Fallback: try to infer emotion from filename
                                    if audio_filename:
                                        filename_lower = audio_filename.lower()
                                        for emotion_key, emotion_value in self.jl_corpus_emotion_map.items():
                                            if emotion_key in filename_lower:
                                                emotion = emotion_value
                                                break
                                    
                                    if not emotion:
                                        emotion = 'Neutral'
                                
                                if not audio_filename:
                                    # Skip if we can't find audio filename
                                    continue
                                
                                # Try to find the actual audio file
                                audio_file_path = None
                                if audio_dir.exists():
                                    # Try exact match first
                                    potential_path = audio_dir / audio_filename
                                    if potential_path.exists():
                                        audio_file_path = potential_path
                                    else:
                                        # Try with different extensions
                                        base_name = Path(audio_filename).stem
                                        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                                            potential_path = audio_dir / f"{base_name}{ext}"
                                            if potential_path.exists():
                                                audio_file_path = potential_path
                                                break
                                
                                # Create training sample
                                if not utterance:
                                    utterance = f"[Audio expression of {emotion.lower()}]"
                                
                                processed_data.append({
                                    'id': str(uuid.uuid4()),
                                    'audio_file': str(audio_file_path) if audio_file_path else None,
                                    'user_message': utterance,
                                    'response': f"I understand you're expressing {emotion.lower()}. That's a valid feeling.",
                                    'context': "JL-Corpus conversation context",
                                    'emotion': emotion,
                                    'audio_features': None,  # Will be processed later if audio exists
                                    'dataset_source': 'jl_corpus',
                                    'timestamp': datetime.datetime.now().isoformat()
                                })

                            except Exception as e:
                                logger.warning(f"Error processing row in {csv_file}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Error processing CSV file {csv_file}: {str(e)}")
                        continue
            
            else:
                # Fallback: Process audio files directly if no CSV found
                logger.info("No CSV files found, processing audio files directly...")
                
                if audio_dir.exists():
                    audio_files = list(audio_dir.glob("*"))
                    audio_files = [f for f in audio_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']]
                    
                    for audio_file in tqdm(audio_files, desc="Processing JL-Corpus audio"):
                        try:
                            # Try to infer emotion from filename
                            filename_lower = audio_file.name.lower()
                            emotion = 'Neutral'
                            
                            for emotion_key, emotion_value in self.jl_corpus_emotion_map.items():
                                if emotion_key in filename_lower:
                                    emotion = emotion_value
                                    break
                            
                            processed_data.append({
                                'id': str(uuid.uuid4()),
                                'audio_file': str(audio_file),
                                'user_message': f"[Audio expression of {emotion.lower()}]",
                                'response': f"I understand you're expressing {emotion.lower()}. That's a valid feeling.",
                                'context': "JL-Corpus audio context",
                                'emotion': emotion,
                                'audio_features': None,
                                'dataset_source': 'jl_corpus',
                                'timestamp': datetime.datetime.now().isoformat()
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error processing {audio_file}: {str(e)}")
                            continue

            logger.info(f"Successfully processed {len(processed_data)} JL-Corpus samples")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing JL-Corpus data: {str(e)}")
            return []

    def create_train_test_val_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/test/val split according to specifications:
        - 70% training: All CREMA-D + 20% from JL-Corpus
        - 20% test: Only from JL-Corpus (remaining data)
        - 10% validation: Only from JL-Corpus (remaining data)
        """
        try:
            logger.info("Creating train/test/validation split...")

            # Get all datasets
            crema_d_data = self.process_crema_d_data()
            jl_corpus_data = self.process_jl_corpus_data()

            logger.info(f"CREMA-D samples: {len(crema_d_data)}")
            logger.info(f"JL-Corpus samples: {len(jl_corpus_data)}")

            # Check if we have JL-Corpus data
            if len(jl_corpus_data) == 0:
                logger.warning("No JL-Corpus data available! Splitting CREMA-D data proportionally...")
                
                # Shuffle CREMA-D data
                crema_shuffled = crema_d_data.copy()
                np.random.shuffle(crema_shuffled)
                
                total_crema = len(crema_shuffled)
                train_size = int(0.7 * total_crema)
                test_size = int(0.2 * total_crema)
                val_size = total_crema - train_size - test_size
                
                # Ensure minimum sizes
                if val_size == 0 and total_crema > 2:
                    train_size -= 1
                    val_size = 1
                if test_size == 0 and total_crema > 1:
                    train_size -= 1
                    test_size = 1
                
                train_data = crema_shuffled[:train_size]
                test_data = crema_shuffled[train_size:train_size + test_size]
                val_data = crema_shuffled[train_size + test_size:]
                
            else:
                # Split JL-Corpus data: 20% for training, 80% for test/val
                jl_train_size = int(0.20 * len(jl_corpus_data))
                jl_remaining = len(jl_corpus_data) - jl_train_size

                # From remaining JL-Corpus: 2/3 for test (20%), 1/3 for val (10%)
                jl_test_size = int((2/3) * jl_remaining)
                jl_val_size = jl_remaining - jl_test_size
                
                # Ensure minimum sizes for JL-Corpus splits
                if jl_val_size == 0 and jl_remaining > 0:
                    jl_test_size -= 1
                    jl_val_size = 1
                
                # Split JL-Corpus data
                jl_shuffled = jl_corpus_data.copy()
                np.random.shuffle(jl_shuffled)

                jl_for_training = jl_shuffled[:jl_train_size]
                jl_for_test = jl_shuffled[jl_train_size:jl_train_size + jl_test_size]
                jl_for_val = jl_shuffled[jl_train_size + jl_test_size:]

                # Create final splits
                train_data = crema_d_data + jl_for_training  # All CREMA-D + 20% JL-Corpus
                test_data = jl_for_test  # 20% of total from JL-Corpus only
                val_data = jl_for_val    # 10% of total from JL-Corpus only
                    
                logger.info(f"Combined dataset split created:")
                logger.info(f"  Training: {len(train_data)} samples (All CREMA-D + {len(jl_for_training)} JL-Corpus)")
                logger.info(f"  Test: {len(test_data)} samples (JL-Corpus only)")
                logger.info(f"  Validation: {len(val_data)} samples (JL-Corpus only)")

            # Final summary
            total_samples = len(train_data) + len(test_data) + len(val_data)
            train_pct = len(train_data) / total_samples * 100
            test_pct = len(test_data) / total_samples * 100
            val_pct = len(val_data) / total_samples * 100

            logger.info(f"Final dataset split:")
            logger.info(f"  Training: {len(train_data)} samples ({train_pct:.1f}%)")
            logger.info(f"  Test: {len(test_data)} samples ({test_pct:.1f}%)")
            logger.info(f"  Validation: {len(val_data)} samples ({val_pct:.1f}%)")

            return train_data, test_data, val_data

        except Exception as e:
            logger.error(f"Error creating train/test/val split: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def setup_datasets(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Main method to setup all datasets"""
        logger.info("Setting up training datasets...")

        # Download datasets
        crema_success = self.download_crema_d()
        jl_success = self.download_jl_corpus()

        if not crema_success:
            logger.warning("CREMA-D download failed, continuing with available data...")

        if not jl_success:
            logger.warning("JL-Corpus download failed, continuing with available data...")

        # Create splits
        return self.create_train_test_val_split()

class AudioProcessor:
    """Handle audio processing for speech-to-text and feature extraction"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.whisper_available = WHISPER_AVAILABLE

        # Initialize Whisper for speech-to-text
        if self.whisper_available and whisper is not None:
            try:
                self.whisper_model = whisper.load_model(config.whisper_model)
                logger.info("✅ Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
                self.whisper_available = False
        else:
            logger.warning("⚠️ Whisper not available - speech-to-text disabled")
            self.whisper_model = None

        # Initialize Wav2Vec2 for audio features
        try:
            self.audio_processor = Wav2Vec2Processor.from_pretrained(config.audio_model)
            self.audio_model = Wav2Vec2Model.from_pretrained(config.audio_model)
            self.audio_model.to(config.device)
            logger.info("✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise

    def convert_audio_format(self, audio_data: bytes, source_format: str) -> np.ndarray:
        """Convert audio from various formats to numpy array"""
        try:
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=self.config.sample_rate)

            # Clean up temporary file
            os.unlink(temp_path)

            return audio

        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            raise

    def extract_audio_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract audio features using Wav2Vec2"""
        try:
            # Ensure audio is the right length
            if len(audio) > self.config.max_audio_length:
                audio = audio[:self.config.max_audio_length]

            # Process audio
            inputs = self.audio_processor(
                audio,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                audio_features = outputs.last_hidden_state  # Shape: [1, seq_len, hidden_size]

            # Global average pooling
            audio_features = audio_features.mean(dim=1)  # Shape: [1, hidden_size]

            return audio_features

        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            raise

    def speech_to_text(self, audio: np.ndarray) -> str:
        """Convert speech to text using Whisper"""
        if not self.whisper_available or self.whisper_model is None:
            logger.warning("Whisper not available, returning placeholder text")
            return "[Audio transcription not available - Whisper not loaded]"

        try:
            result = self.whisper_model.transcribe(audio)
            return result["text"].strip()

        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {str(e)}")
            return "[Transcription failed]"

    def process_audio_file(self, audio_data: bytes, file_extension: str) -> Dict:
        """Complete audio processing pipeline"""
        try:
            # Convert audio to numpy array
            audio = self.convert_audio_format(audio_data, file_extension.lstrip('.'))

            # Extract text
            transcribed_text = self.speech_to_text(audio)

            # Extract audio features
            audio_features = self.extract_audio_features(audio)

            return {
                'transcribed_text': transcribed_text,
                'audio_features': audio_features,
                'audio_length': len(audio) / self.config.sample_rate,
                'sample_rate': self.config.sample_rate
            }

        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise

class ConversationDataset(Dataset):
    """Dataset class for conversation data with emotion labels and audio features"""

    def __init__(self, conversations: List[Dict], tokenizer, config: ModelConfig):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.config = config
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(config.emotion_classes)}

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Prepare input text (conversation context + current user message)
        context = conversation.get('context', '')
        user_message = conversation.get('user_message', '')
        response = conversation.get('response', '')
        emotion_label = conversation.get('emotion', 'Neutral')

        # Combine context and user message
        input_text = f"{context} [SEP] {user_message}"

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )

        # Tokenize response
        response_encoding = self.tokenizer(
            response,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )

        # Convert emotion to ID
        emotion_id = self.emotion_to_id.get(emotion_label, self.emotion_to_id['Neutral'])

        # Handle audio features if available
        audio_features = conversation.get('audio_features')
        if audio_features is not None:
            if isinstance(audio_features, torch.Tensor):
                audio_features = audio_features.squeeze()
            else:
                # Create dummy audio features if not available
                audio_features = torch.zeros(self.config.audio_hidden_size)
        else:
            audio_features = torch.zeros(self.config.audio_hidden_size)

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'response_ids': response_encoding['input_ids'].squeeze(),
            'response_attention_mask': response_encoding['attention_mask'].squeeze(),
            'emotion_label': torch.tensor(emotion_id, dtype=torch.long),
            'audio_features': audio_features,
            'conversation_id': conversation.get('id', ''),
            'timestamp': conversation.get('timestamp', '')
        }

class EmotionalIntelligenceModel(nn.Module):
    """Neural network model for emotion-aware conversation generation with audio support"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.transformer = AutoModel.from_pretrained(config.base_model)
        
        # Get the actual hidden size from the loaded model
        actual_hidden_size = self.transformer.config.hidden_size
        print(f"Actual model hidden size: {actual_hidden_size}")
        
        # Update config if needed
        if actual_hidden_size != config.hidden_size:
            print(f"Warning: Config hidden_size ({config.hidden_size}) != actual hidden_size ({actual_hidden_size})")
            print(f"Using actual hidden_size: {actual_hidden_size}")
            config.hidden_size = actual_hidden_size
            
        # Audio feature projection layer
        self.audio_projection = nn.Linear(config.audio_hidden_size, config.hidden_size)

        # Multimodal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

        # Emotion classification head (enhanced with audio)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, len(config.emotion_classes))
        )

        # Response generation head
        self.response_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, self.tokenizer.vocab_size)
        )

        # Emotion-conditioned attention
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate
        )

    def forward(self, input_ids, attention_mask, audio_features=None, response_ids=None, training=True):
        #Ensure all inputs are on the same device
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        if audio_features is not None:
            audio_features = audio_features.to(device)
            
        if response_ids is not None:
            response_ids = response_ids.to(device)
            
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states
        hidden_states = transformer_outputs.last_hidden_state
        text_pooled = hidden_states.mean(dim=1)  # Global average pooling

        # Process audio features if available
        if audio_features is not None:
            audio_projected = self.audio_projection(audio_features)

            # Multimodal fusion
            combined_features = torch.cat([text_pooled, audio_projected], dim=-1)
            fused_output = self.fusion_layer(combined_features)
        else:
            fused_output = text_pooled

        # Emotion classification
        emotion_logits = self.emotion_classifier(fused_output)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        outputs = {
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'hidden_states': hidden_states,
            'fused_output': fused_output
        }

        # Response generation (during training or inference)
        if response_ids is not None or not training:
            # Apply emotion-conditioned attention
            emotion_context = fused_output.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
            attended_states, _ = self.emotion_attention(
                hidden_states.transpose(0, 1),
                emotion_context.transpose(0, 1),
                emotion_context.transpose(0, 1)
            )
            attended_states = attended_states.transpose(0, 1)

            # Generate response logits
            response_logits = self.response_generator(attended_states)
            outputs['response_logits'] = response_logits

        return outputs
    
    def generate_response(self, input_text: str, audio_features: torch.Tensor = None,
                         max_length: int = 100, temperature: float = 0.7):
        #Move inputs to the correct device
        inputs={k:v.to(device) for k,v in inputs.item()}
                             
        #Ensure transformer is on the corect device
        self.transformer=self.transform.to(device)
        """Generate an emotionally aware response with optional audio input"""
        self.eval()

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length
            ).to(self.config.device)

            # Prepare audio features
            if audio_features is not None:
                audio_features = audio_features.to(self.config.device)
                if audio_features.dim() == 1:
                    audio_features = audio_features.unsqueeze(0)

            # Get model outputs
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                audio_features=audio_features,
                training=False
            )

            # Detect emotion
            emotion_probs = outputs['emotion_probs']
            predicted_emotion_id = torch.argmax(emotion_probs, dim=-1).item()
            predicted_emotion = self.config.emotion_classes[predicted_emotion_id]
            emotion_confidence = emotion_probs[0][predicted_emotion_id].item()

            # Generate response using the transformer's generation method
            generated = self.transformer.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Decode response
            response = self.tokenizer.decode(
                generated[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return {
                'response': response.strip(),
                'detected_emotion': predicted_emotion,
                'emotion_confidence': emotion_confidence,
                'emotion_distribution': {
                    emotion: prob.item()
                    for emotion, prob in zip(self.config.emotion_classes, emotion_probs[0])
                },
                'has_audio_input': audio_features is not None
            }
    def to(self,device): 
        """Override to method to ensure all components are moved to the same device"""
        super().to(device)
        if hasattr(self,'transformer'):
            self.transformer=self.transformer.to(device)
        return self

class EmotionalAITrainer:
    """Training pipeline for the emotional intelligence model with integrated datasets"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = EmotionalIntelligenceModel(config).to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.response_criterion = nn.CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
        self.audio_processor = AudioProcessor(config)

    def preprocess_dataset_audio_features(self, dataset: List[Dict]) -> List[Dict]:
        """Preprocess audio features for dataset samples"""
        processed_dataset = []

        for sample in tqdm(dataset, desc="Processing audio features"):
            try:
                # Copy sample
                processed_sample = sample.copy()

                # Process audio if file exists
                if sample.get('audio_file') and os.path.exists(sample['audio_file']):
                    try:
                        # Load audio
                        audio, sr = librosa.load(sample['audio_file'], sr=self.config.sample_rate)

                        # Extract audio features using our processor
                        audio_features = self.audio_processor.extract_audio_features(audio)
                        processed_sample['audio_features'] = audio_features

                    except Exception as e:
                        logger.warning(f"Could not process audio {sample['audio_file']}: {str(e)}")
                        # Create dummy features if audio processing fails
                        processed_sample['audio_features'] = torch.zeros(self.config.audio_hidden_size)
                else:
                    # Create dummy features for samples without audio
                    processed_sample['audio_features'] = torch.zeros(self.config.audio_hidden_size)

                processed_dataset.append(processed_sample)

            except Exception as e:
                logger.warning(f"Error preprocessing sample: {str(e)}")
                continue

        return processed_dataset

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        emotion_correct = 0
        total_samples = 0
        
        # Initialize gradient accumulation
        accumulation_steps = self.config.gradient_accumulation_steps
        accumulation_counter = 0
        
        # Calculate effective batch size
        effective_batch_size = self.config.batch_size * accumulation_steps
        logger.info(f"Training with gradient accumulation: {accumulation_steps} steps, "
                   f"effective batch size: {effective_batch_size}")
        
        # Zero gradients at the start
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            response_ids = batch['response_ids'].to(self.config.device)
            emotion_labels = batch['emotion_label'].to(self.config.device)
            audio_features = batch['audio_features'].to(self.config.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                response_ids=response_ids,
                training=True
            )

            # Calculate losses
            emotion_loss = self.emotion_criterion(outputs['emotion_logits'], emotion_labels)

            # Response generation loss (teacher forcing)
            if 'response_logits' in outputs:
                response_loss = self.response_criterion(
                    outputs['response_logits'].view(-1, outputs['response_logits'].size(-1)),
                    response_ids.view(-1)
                )
                total_loss_batch = emotion_loss + response_loss
            else:
                total_loss_batch = emotion_loss
                
            # Scale loss by accumulation steps
            scaled_loss = total_loss_batch / accumulation_steps
            
            # Backward pass (accumulate gradients)
            scaled_loss.backward()
            
            # Update accumulation counter
            accumulation_counter += 1
            
            # Check if we should update parameters
            is_last_batch = (batch_idx + 1) == len(dataloader)
            should_update = (accumulation_counter % accumulation_steps == 0) or is_last_batch
            
            if should_update:
                # Gradient clipping (optional, but recommended)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Reset accumulation counter if not end of epoch
                if not is_last_batch:
                    accumulation_counter = 0

            # Track metrics (use original unscaled loss for tracking)
            total_loss += total_loss_batch.item()
            emotion_pred = torch.argmax(outputs['emotion_logits'], dim=-1)
            emotion_correct += (emotion_pred == emotion_labels).sum().item()
            total_samples += len(emotion_labels)

        avg_loss = total_loss / len(dataloader)
        emotion_accuracy = emotion_correct / total_samples

        return avg_loss, emotion_accuracy

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        emotion_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                emotion_labels = batch['emotion_label'].to(self.config.device)
                audio_features = batch['audio_features'].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_features=audio_features,
                    training=False
                )

                emotion_loss = self.emotion_criterion(outputs['emotion_logits'], emotion_labels)
                total_loss += emotion_loss.item()

                emotion_pred = torch.argmax(outputs['emotion_logits'], dim=-1)
                emotion_correct += (emotion_pred == emotion_labels).sum().item()
                total_samples += len(emotion_labels)

                all_predictions.extend(emotion_pred.cpu().numpy())
                all_labels.extend(emotion_labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        emotion_accuracy = emotion_correct / total_samples

        return avg_loss, emotion_accuracy, all_predictions, all_labels

    def train_with_integrated_datasets(self):
        """Train model using CREMA-D and JL-Corpus datasets"""
        logger.info("Starting training with integrated datasets...")

        # Setup datasets
        dataset_downloader = DatasetDownloader(self.config)
        train_data, test_data, val_data = dataset_downloader.setup_datasets()

        if not train_data:
            logger.error("No training data available!")
            return None

        # Preprocess audio features
        logger.info("Preprocessing audio features...")
        train_data = self.preprocess_dataset_audio_features(train_data)
        test_data = self.preprocess_dataset_audio_features(test_data)
        val_data = self.preprocess_dataset_audio_features(val_data)

        # Create data loaders
        train_dataset = ConversationDataset(train_data, self.model.tokenizer, self.config)
        val_dataset = ConversationDataset(val_data, self.model.tokenizer, self.config)
        test_dataset = ConversationDataset(test_data, self.model.tokenizer, self.config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Training loop
        best_val_accuracy = 0
        training_history = []

        logger.info(f"Starting training with {len(train_data)} training samples...")

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)

            # Validation
            val_loss, val_accuracy, val_preds, val_labels = self.validate(val_loader)

            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model("best_emotional_ai_model.pt")
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

        # Final test evaluation
        logger.info("Evaluating on test set...")
        test_loss, test_accuracy, test_preds, test_labels = self.validate(test_loader)
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        # Generate classification report
        try:
            from sklearn.metrics import classification_report
            emotion_names = [self.config.emotion_classes[i] for i in range(len(self.config.emotion_classes))]
            report = classification_report(test_labels, test_preds, target_names=emotion_names)
            logger.info(f"Classification Report:\n{report}")
        except Exception as e:
            logger.warning(f"Could not generate classification report: {str(e)}")

        # Save training results
        results = {
            'training_history': training_history,
            'final_test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'dataset_info': {
                'total_train_samples': len(train_data),
                'total_val_samples': len(val_data),
                'total_test_samples': len(test_data),
                'crema_d_samples': len([d for d in train_data if d.get('dataset_source') == 'crema_d']),
                'jl_train_samples': len([d for d in train_data if d.get('dataset_source') == 'jl_corpus']),
                'jl_test_samples': len([d for d in test_data if d.get('dataset_source') == 'jl_corpus']),
                'jl_val_samples': len([d for d in val_data if d.get('dataset_source') == 'jl_corpus'])
            }
        }

        # Save results to file
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Training completed! Results saved to training_results.json")
        return results

    def train(self, train_conversations, val_conversations):
        """Full training pipeline for custom data"""
        # Create datasets
        train_dataset = ConversationDataset(train_conversations, self.model.tokenizer, self.config)
        val_dataset = ConversationDataset(val_conversations, self.model.tokenizer, self.config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        best_val_accuracy = 0
        training_history = []

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)

            # Validation
            val_loss, val_accuracy, val_preds, val_labels = self.validate(val_loader)

            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model("best_emotional_ai_model.pt")
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

        return training_history

    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class ConversationDatabase:
    """Database management for conversations and ratings"""

    def __init__(self, db_path: str = "emotional_ai.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table - updated to include audio info
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                context TEXT,
                detected_emotion TEXT,
                emotion_confidence REAL,
                audio_filename TEXT,
                audio_length REAL,
                has_audio_input BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Ratings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_rating INTEGER NOT NULL,
                feedback_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        # Poor performance queue for annotation
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotation_queue (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_rating INTEGER NOT NULL,
                annotation_status TEXT DEFAULT 'pending',
                annotated_emotion TEXT,
                annotator_id TEXT,
                annotation_timestamp DATETIME,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        conn.commit()
        conn.close()

    def save_conversation(self, conversation_data: Dict) -> str:
        """Save a conversation to the database"""
        conversation_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations
            (id, user_message, ai_response, context, detected_emotion, emotion_confidence,
             audio_filename, audio_length, has_audio_input)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_id,
            conversation_data['user_message'],
            conversation_data['ai_response'],
            conversation_data.get('context', ''),
            conversation_data.get('detected_emotion', ''),
            conversation_data.get('emotion_confidence', 0.0),
            conversation_data.get('audio_filename', ''),
            conversation_data.get('audio_length', 0.0),
            conversation_data.get('has_audio_input', False)
        ))

        conn.commit()
        conn.close()

        return conversation_id

    def save_rating(self, conversation_id: str, rating: int, feedback: str = "") -> str:
        """Save user rating and handle poor performance cases"""
        rating_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save rating
        cursor.execute('''
            INSERT INTO ratings (id, conversation_id, user_rating, feedback_text)
            VALUES (?, ?, ?, ?)
        ''', (rating_id, conversation_id, rating, feedback))

        # If rating is 1 or 2, add to annotation queue
        if rating <= 2:
            annotation_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO annotation_queue (id, conversation_id, user_rating)
                VALUES (?, ?, ?)
            ''', (annotation_id, conversation_id, rating))

        conn.commit()
        conn.close()

        return rating_id

    def get_conversations_for_annotation(self, limit: int = 50) -> List[Dict]:
        """Get conversations that need annotation (ratings 1-2)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                aq.id as annotation_id,
                c.id as conversation_id,
                c.user_message,
                c.ai_response,
                c.context,
                c.detected_emotion,
                c.emotion_confidence,
                aq.user_rating,
                aq.annotation_status
            FROM annotation_queue aq
            JOIN conversations c ON aq.conversation_id = c.id
            WHERE aq.annotation_status = 'pending'
            ORDER BY c.timestamp DESC
            LIMIT ?
        ''', (limit,))

        results = cursor.fetchall()
        conn.close()

        conversations = []
        for row in results:
            conversations.append({
                'annotation_id': row[0],
                'conversation_id': row[1],
                'user_message': row[2],
                'ai_response': row[3],
                'context': row[4],
                'detected_emotion': row[5],
                'emotion_confidence': row[6],
                'user_rating': row[7],
                'annotation_status': row[8]
            })

        return conversations

    def update_annotation(self, annotation_id: str, annotated_emotion: str, annotator_id: str):
        """Update annotation with human-labeled emotion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE annotation_queue
            SET annotation_status = 'completed',
                annotated_emotion = ?,
                annotator_id = ?,
                annotation_timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (annotated_emotion, annotator_id, annotation_id))

        conn.commit()
        conn.close()

    def get_training_data(self) -> List[Dict]:
        """Get annotated conversations for model retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                c.user_message,
                c.ai_response,
                c.context,
                COALESCE(aq.annotated_emotion, c.detected_emotion) as emotion,
                c.id
            FROM conversations c
            LEFT JOIN annotation_queue aq ON c.id = aq.conversation_id
                AND aq.annotation_status = 'completed'
            WHERE c.detected_emotion IS NOT NULL
        ''')

        results = cursor.fetchall()
        conn.close()

        training_data = []
        for row in results:
            training_data.append({
                'user_message': row[0],
                'response': row[1],
                'context': row[2],
                'emotion': row[3],
                'id': row[4]
            })

        return training_data

# Initialize database
db = ConversationDatabase()

# === Notebook Helper Functions ===

def train_emotional_ai():
    """
    Main function to call from Jupyter notebook
    Usage in notebook:
        from emotional_ai_backend import train_emotional_ai
        results = train_emotional_ai()
    """
    print("🧠 Emotional AI Training - Notebook Mode")
    print("=" * 50)

    try:
        # Setup logging for notebook
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

        # Configuration
        config = ModelConfig()
        print(f"📱 Device: {config.device}")
        print(f"🎭 Emotion Classes: {config.emotion_classes}")
        print(f"📊 Batch Size: {config.batch_size}")
        print(f"🔄 Epochs: {config.num_epochs}")
        print()

        # Check for required dependencies
        missing_deps = []
        if not WHISPER_AVAILABLE:
            missing_deps.append("OpenAI Whisper")
        if not KAGGLEHUB_AVAILABLE:
            missing_deps.append("Kagglehub")
            
        if missing_deps:
            print(f"⚠️ Warning: Some dependencies are missing: {', '.join(missing_deps)}")
            print("   Training will continue but some features may be limited.")

        # Initialize trainer
        print("🔧 Initializing trainer...")
        trainer = EmotionalAITrainer(config)

        # Run training
        print("🚀 Starting training with integrated datasets...")
        results = trainer.train_with_integrated_datasets()

        if results:
            print("\n🎉 Training Completed Successfully!")
            print("=" * 50)
            print(f"📈 Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
            print(f"🎯 Final Test Accuracy: {results['final_test_accuracy']:.4f}")
            print(f"📊 Dataset Statistics:")
            print(f"   • Total Training Samples: {results['dataset_info']['total_train_samples']:,}")
            print(f"   • CREMA-D Samples: {results['dataset_info']['crema_d_samples']:,}")
            print(f"   • JL-Corpus Train Samples: {results['dataset_info']['jl_train_samples']:,}")
            print(f"   • Test Samples: {results['dataset_info']['total_test_samples']:,}")
            print(f"   • Validation Samples: {results['dataset_info']['total_val_samples']:,}")

            # Plot training history if matplotlib is available
            try:
                import matplotlib.pyplot as plt

                history = results['training_history']
                epochs = [h['epoch'] for h in history]
                train_acc = [h['train_accuracy'] for h in history]
                val_acc = [h['val_accuracy'] for h in history]

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
                plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Emotional AI Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

            except ImportError:
                print("📊 Install matplotlib to see training plots: pip install matplotlib")

            return results
        else:
            print("❌ Training failed!")
            return None

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def quick_test_model(model_path: str = "best_emotional_ai_model.pt"):
    """
    Quick test function for the trained model
    Usage in notebook:
        from emotional_ai_backend import quick_test_model
        quick_test_model()
    """
    try:
        config = ModelConfig()
        model = EmotionalIntelligenceModel(config)

        # Load trained model
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Test with sample text
        test_messages = [
            "I'm so happy today!",
            "I feel really sad and down",
            "This is making me angry",
            "I'm scared about what might happen",
            "What a pleasant surprise!"
        ]

        print("🧪 Testing Trained Model")
        print("=" * 40)

        for message in test_messages:
            result = model.generate_response(message)
            print(f"Input: {message}")
            print(f"Detected Emotion: {result['detected_emotion']} ({result['emotion_confidence']:.2f})")
            print(f"Response: {result['response']}")
            print("-" * 40)

        return True

    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

def setup_kaggle_credentials():
    """
    Helper function to setup Kaggle credentials for dataset download
    Usage in notebook:
        from emotional_ai_backend import setup_kaggle_credentials
        setup_kaggle_credentials()
    """
    print("🔑 Kaggle Setup Instructions")
    print("=" * 30)
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Scroll to 'Create New API Token'")
    print("3. Download kaggle.json file")
    print("4. Upload kaggle.json to your notebook environment")
    print("5. Run the next cell")

def download_datasets_only():
    """
    Download datasets without training (useful for data exploration)
    Usage in notebook:
        from emotional_ai_backend import download_datasets_only
        download_datasets_only()
    """
    try:
        config = ModelConfig()
        downloader = DatasetDownloader(config)

        print("📥 Downloading datasets...")
        crema_success = downloader.download_crema_d()
        jl_success = downloader.download_jl_corpus()

        if crema_success:
            print("✅ CREMA-D dataset downloaded successfully")
        else:
            print("❌ CREMA-D download failed")

        if jl_success:
            print("✅ JL-Corpus dataset setup completed")
        else:
            print("❌ JL-Corpus setup failed")

        return crema_success and jl_success

    except Exception as e:
        print(f"❌ Dataset download failed: {str(e)}")
        return False

def explore_datasets():
    """
    Explore downloaded datasets without training
    Usage in notebook:
        from emotional_ai_backend import explore_datasets
        explore_datasets()
    """
    try:
        config = ModelConfig()
        downloader = DatasetDownloader(config)

        print("🔍 Exploring Datasets")
        print("=" * 25)

        # Process datasets
        crema_data = downloader.process_crema_d_data()
        jl_data = downloader.process_jl_corpus_data()

        print(f"📊 CREMA-D Dataset: {len(crema_data)} samples")
        if crema_data:
            emotions = [d['emotion'] for d in crema_data]
            emotion_counts = pd.Series(emotions).value_counts()
            print("   Emotion Distribution:")
            for emotion, count in emotion_counts.items():
                print(f"     {emotion}: {count}")

        print(f"\n📊 JL-Corpus Dataset: {len(jl_data)} samples")
        if jl_data:
            emotions = [d['emotion'] for d in jl_data]
            emotion_counts = pd.Series(emotions).value_counts()
            print("   Emotion Distribution:")
            for emotion, count in emotion_counts.items():
                print(f"     {emotion}: {count}")

        # Create split preview
        train_data, test_data, val_data = downloader.create_train_test_val_split()

        total_samples = len(train_data) + len(test_data) + len(val_data)
        print(f"\n📋 Proposed Data Split:")
        print(f"   Training: {len(train_data):,} samples ({len(train_data)/total_samples*100:.1f}%)")
        print(f"   Testing: {len(test_data):,} samples ({len(test_data)/total_samples*100:.1f}%)")
        print(f"   Validation: {len(val_data):,} samples ({len(val_data)/total_samples*100:.1f}%)")

        return {
            'crema_data': crema_data,
            'jl_data': jl_data,
            'train_data': train_data,
            'test_data': test_data,
            'val_data': val_data
        }

    except Exception as e:
        print(f"❌ Dataset exploration failed: {str(e)}")
        return None

def prepare_training_data_from_csv(csv_path: str, audio_dir: str = None) -> List[Dict]:
    """
    Prepare training data from CSV file with optional audio features
    Expected columns: user_message, response, context, emotion, audio_filename (optional)
    """
    df = pd.read_csv(csv_path)

    # Initialize audio processor if audio directory is provided
    if audio_dir and os.path.exists(audio_dir):
        config = ModelConfig()
        processor = AudioProcessor(config)
    else:
        processor = None

    training_data = []
    for _, row in df.iterrows():
        conversation = {
            'user_message': str(row['user_message']),
            'response': str(row['response']),
            'context': str(row.get('context', '')),
            'emotion': str(row['emotion']),
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Process audio if available
        if processor and 'audio_filename' in row and pd.notna(row['audio_filename']):
            audio_path = os.path.join(audio_dir, str(row['audio_filename']))
            if os.path.exists(audio_path):
                try:
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()

                    file_ext = os.path.splitext(audio_path)[1]
                    audio_result = processor.process_audio_file(audio_data, file_ext)
                    conversation['audio_features'] = audio_result['audio_features']

                except Exception as e:
                    logger.warning(f"Could not process audio file {audio_path}: {str(e)}")

        training_data.append(conversation)

    return training_data

def export_annotation_data_for_audio_app(output_path: str):
    """
    Export poor performance conversations for the audio annotation app
    This creates CSV files compatible with your existing annotation system
    """
    conversations = db.get_conversations_for_annotation(limit=1000)

    # Prepare data for the audio annotation app format
    annotation_data = []
    for conv in conversations:
        # Use existing audio filename or create placeholder
        audio_filename = conv.get('audio_filename', f"conversation_{conv['conversation_id']}.wav")

        annotation_data.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'audio_filename': audio_filename,
            'conversation_id': conv['conversation_id'],
            'user_message': conv['user_message'],
            'ai_response': conv['ai_response'],
            'detected_emotion': conv['detected_emotion'],
            'user_rating': conv['user_rating'],
            'annotation_status': conv['annotation_status'],
            'rater_id': 'AI_System',  # Compatible with your annotation app
            'emotion_1': conv['detected_emotion'],  # Primary emotion for Fleiss' Kappa
            'emotion_2': None,
            'emotion_3': None,
            'intensity_level': 2,  # Default medium intensity
            'confidence_score': 3   # Default moderate confidence
        })

    # Save to CSV in format compatible with your annotation app
    df = pd.DataFrame(annotation_data)
    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(annotation_data)} conversations for annotation to {output_path}")

def create_sample_training_data():
    """Create sample training data for testing purposes"""
    print("📝 Creating Sample Training Data")
    print("=" * 35)

    sample_conversations = [
        {
            'id': str(uuid.uuid4()),
            'user_message': "I'm feeling really happy today!",
            'response': "That's wonderful to hear! I'm glad you're having such a positive day. What's making you feel so happy?",
            'context': "User expressing positive emotions",
            'emotion': 'Joy',
            'audio_features': None,
            'timestamp': datetime.datetime.now().isoformat()
        },
        {
            'id': str(uuid.uuid4()),
            'user_message': "I'm worried about my job interview tomorrow",
            'response': "It's completely natural to feel nervous before an important interview. Remember that they invited you because they're interested in you. Try to focus on your strengths and prepare some questions about the role.",
            'context': "User expressing anxiety about future event",
            'emotion': 'Fear',
            'audio_features': None,
            'timestamp': datetime.datetime.now().isoformat()
        },
        {
            'id': str(uuid.uuid4()),
            'user_message': "I can't believe they cancelled my vacation",
            'response': "I can understand how frustrating and disappointing that must be, especially when you were looking forward to it. Have you been able to find out why it was cancelled or if there might be alternative dates?",
            'context': "User expressing anger about cancelled plans",
            'emotion': 'Anger',
            'audio_features': None,
            'timestamp': datetime.datetime.now().isoformat()
        },
        {
            'id': str(uuid.uuid4()),
            'user_message': "I feel so sad after watching that movie",
            'response': "Movies can really affect our emotions, especially when they touch on meaningful themes. It sounds like the film resonated with you. Sometimes a good cry can actually be cathartic.",
            'context': "User expressing sadness from media",
            'emotion': 'Sadness',
            'audio_features': None,
            'timestamp': datetime.datetime.now().isoformat()
        },
        {
            'id': str(uuid.uuid4()),
            'user_message': "Wow, I never expected to win the lottery!",
            'response': "Congratulations! That must be such an incredible and surprising feeling. Winning the lottery is definitely something most people never expect to happen to them.",
            'context': "User expressing surprise about unexpected event",
            'emotion': 'Surprise',
            'audio_features': None,
            'timestamp': datetime.datetime.now().isoformat()
        }
    ]

    print(f"✅ Created {len(sample_conversations)} sample conversations")
    print("Emotion distribution:")
    emotions = [conv['emotion'] for conv in sample_conversations]
    emotion_counts = pd.Series(emotions).value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")

    return sample_conversations

def train_with_sample_data():
    """Train model with sample data for testing"""
    try:
        print("🧪 Training with Sample Data")
        print("=" * 35)

        # Create sample data
        sample_data = create_sample_training_data()

        # Split data
        train_size = int(0.8 * len(sample_data))
        train_data = sample_data[:train_size]
        val_data = sample_data[train_size:]

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        # Initialize trainer
        config = ModelConfig()
        config.num_epochs = 3  # Reduced for quick testing
        config.batch_size = 2   # Small batch for sample data

        trainer = EmotionalAITrainer(config)

        # Train
        print("🚀 Starting training...")
        history = trainer.train(train_data, val_data)

        print("✅ Sample training completed!")
        return history

    except Exception as e:
        print(f"❌ Sample training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# FastAPI Application
app = FastAPI(title="Emotional Intelligence LLM API")

# Global model instance
model = None

# API Models
class ConversationRequest(BaseModel):
    user_message: str
    context: str = ""
    session_id: str = ""

class ConversationResponse(BaseModel):
    conversation_id: str
    ai_response: str
    detected_emotion: str
    emotion_confidence: float
    emotion_distribution: Dict[str, float]

class RatingRequest(BaseModel):
    conversation_id: str
    rating: int
    feedback: str = ""

class AnnotationRequest(BaseModel):
    annotation_id: str
    annotated_emotion: str
    annotator_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    config = ModelConfig()

    try:
        # Try to load existing model
        trainer = EmotionalAITrainer(config)
        trainer.load_model("best_emotional_ai_model.pt")
        model = trainer.model
        logger.info("Loaded existing trained model")
    except FileNotFoundError:
        # Initialize new model if no trained model exists
        model = EmotionalIntelligenceModel(config)
        logger.info("Initialized new model - training required")

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Main chat endpoint for conversation"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Prepare input text with context
        input_text = f"{request.context} {request.user_message}".strip()

        # Generate response
        result = model.generate_response(input_text)

        # Save conversation to database
        conversation_data = {
            'user_message': request.user_message,
            'ai_response': result['response'],
            'context': request.context,
            'detected_emotion': result['detected_emotion'],
            'emotion_confidence': result['emotion_confidence']
        }

        conversation_id = db.save_conversation(conversation_data)

        return ConversationResponse(
            conversation_id=conversation_id,
            ai_response=result['response'],
            detected_emotion=result['detected_emotion'],
            emotion_confidence=result['emotion_confidence'],
            emotion_distribution=result['emotion_distribution']
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate")
async def rate_conversation(request: RatingRequest):
    """Rate a conversation (1-5 scale)"""
    try:
        rating_id = db.save_rating(
            request.conversation_id,
            request.rating,
            request.feedback
        )

        message = "Rating saved successfully"
        if request.rating <= 2:
            message += " - Added to annotation queue for review"

        return {"rating_id": rating_id, "message": message}

    except Exception as e:
        logger.error(f"Error in rating endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/annotation-queue")
async def get_annotation_queue(limit: int = 50):
    """Get conversations that need human annotation"""
    try:
        conversations = db.get_conversations_for_annotation(limit)
        return {"conversations": conversations, "count": len(conversations)}

    except Exception as e:
        logger.error(f"Error getting annotation queue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/annotate")
async def annotate_conversation(request: AnnotationRequest):
    """Submit human annotation for a conversation"""
    try:
        db.update_annotation(
            request.annotation_id,
            request.annotated_emotion,
            request.annotator_id
        )

        return {"message": "Annotation saved successfully"}

    except Exception as e:
        logger.error(f"Error in annotation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

# === Main execution logic ===
if __name__ == "__main__":
    import uvicorn

    # Enhanced training setup with integrated datasets
    def run_integrated_training():
        """Run training with CREMA-D and JL-Corpus datasets"""
        logger.info("🧠 Starting Emotional AI Training with Integrated Datasets")

        # Configuration
        config = ModelConfig()

        # Initialize trainer
        trainer = EmotionalAITrainer(config)

        # Run training with integrated datasets
        results = trainer.train_with_integrated_datasets()

        if results:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"📊 Final Results:")
            logger.info(f"   - Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
            logger.info(f"   - Final Test Accuracy: {results['final_test_accuracy']:.4f}")
            logger.info(f"   - Total Training Samples: {results['dataset_info']['total_train_samples']}")
            logger.info(f"   - CREMA-D Samples: {results['dataset_info']['crema_d_samples']}")
            logger.info(f"   - JL-Corpus Train Samples: {results['dataset_info']['jl_train_samples']}")
        else:
            logger.error("❌ Training failed!")

    # Example usage for different environments
    if os.getenv("JUPYTER_NOTEBOOK", "false").lower() == "true":
        # Notebook environment
        print("📓 Running in Jupyter notebook mode")
        results = train_emotional_ai()
    elif os.getenv("RUN_TRAINING", "false").lower() == "true":
        # Training mode
        run_integrated_training()
    else:
        # API server mode
        print("🌐 Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
