# -*- coding: utf-8 -*-
"""
Emotional Intelligence AI 
This code generates an emotionally intelligent LLM. It uses multiple techniques including the following: 
    1. Base model: uses roberta-large, which significantly increases the capacity to learn complex patterns
    2. Adversarial training (FGM): adds a tiny calculated "attack" and trains the model to resistant the model will learn more generalizable features
    3. Label smoothing: less confidence in the prediction, which acts as a regularizer
    4. Audio augmentation: makes the model more resilient to different audio recording conditions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import json
import datetime
import uuid
import os
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                           accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

# Handle imports
def safe_import(package_name, install_command=None):
    try:
        if package_name == "kagglehub":
            import kagglehub
            return kagglehub, True
        elif package_name == "whisper":
            import whisper
            return whisper, True
        elif package_name == "librosa":
            import librosa
            return librosa, True
        elif package_name == "soundfile":
            import soundfile
            return soundfile, True
    except ImportError:
        if install_command:
            print(f"Installing {package_name}...")
            import subprocess
            try:
                subprocess.check_call(install_command)
                if package_name == "kagglehub":
                    import kagglehub
                    return kagglehub, True
                elif package_name == "whisper":
                    import whisper
                    return whisper, True
                elif package_name == "librosa":
                    import librosa
                    return librosa, True
                elif package_name == "soundfile":
                    import soundfile
                    return soundfile, True
            except Exception as e:
                print(f"Could not install {package_name}: {e}")
        return None, False

# Try imports
kagglehub, KAGGLEHUB_AVAILABLE = safe_import("kagglehub", ["pip", "install", "kagglehub"])
whisper, WHISPER_AVAILABLE = safe_import("whisper", ["pip", "install", "openai-whisper"])
librosa, LIBROSA_AVAILABLE = safe_import("librosa", ["pip", "install", "librosa"])
soundfile, SOUNDFILE_AVAILABLE = safe_import("soundfile", ["pip", "install", "soundfile"])

print(f"Dependencies - Kagglehub: {KAGGLEHUB_AVAILABLE}, Whisper: {WHISPER_AVAILABLE}, Librosa: {LIBROSA_AVAILABLE}")

@dataclass
class EnhancedConfig:
    """Enhanced configuration for 40% target performance"""
    # Strategy: Upgrade to a larger, more capable model
    base_model: str = "roberta-large"
    hidden_size: int = 1024 # Corresponds to roberta-large

    # Model parameters
    emotion_classes: List[str] = None
    max_sequence_length: int = 256
    dropout_rate: float = 0.1

    # Enhanced training parameters
    learning_rate: float = 1e-5
    batch_size: int = 8  # Reduced batch size for larger model
    num_epochs: int = 30
    warmup_ratio: float = 0.2
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Data split strategy
    audio_test_ratio: float = 0.10
    audio_val_ratio: float = 0.10

    # Synthetic data strategy
    synthetic_samples_per_emotion: int = 800
    use_domain_adaptation: bool = True

    # Audio processing strategy
    whisper_model_size: str = "large"
    use_multiple_transcriptions: bool = True
    audio_quality_threshold: float = 0.6

    # Advanced training techniques
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    audio_weight: float = 3.0
    use_mixup: bool = True
    mixup_alpha: float = 0.2

    # NEW STRATEGY: Label Smoothing for regularization
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1

    # NEW STRATEGY: Adversarial Training for robustness
    use_adversarial_training: bool = True

    # NEW STRATEGY: Audio Augmentation
    use_audio_augmentation: bool = True
    audio_noise_factor: float = 0.005
    audio_pitch_shift_steps: int = 2

    # Model architecture improvements
    use_multi_layer_pooling: bool = True
    use_attention_pooling: bool = True
    hidden_layers: List[int] = None

    # Advanced optimization
    use_cosine_schedule: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    # Other parameters
    use_audio_transcription: bool = True
    sample_rate: int = 16000
    max_audio_length: float = 12.0
    min_text_length: int = 2
    use_audio_preprocessing: bool = True

    # Dataset paths
    crema_dataset_path: str = "./crema_d_data"
    ravdess_dataset_path: str = "./ravdess_data"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.emotion_classes is None:
            self.emotion_classes = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
        if self.hidden_layers is None:
            self.hidden_layers = [1024, 512, 256]

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, classes, smoothing=0.0, dim=-1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * pred, dim=self.dim)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedSyntheticGenerator:
    """Enhanced synthetic data generation with domain adaptation"""

    def __init__(self, config: EnhancedConfig):
        self.config = config

        # Strategy: Make synthetic data more like transcribed speech
        self.audio_like_templates = {
            'Anger': [
                "I'm angry about this", "This makes me mad", "I hate this",
                "I'm so frustrated", "This is annoying", "I'm upset about this",
                "This bothers me", "I don't like this", "This makes me furious",
                "I'm irritated by this"
            ],
            'Disgust': [
                "This is gross", "I find this disgusting", "This is nasty",
                "This makes me sick", "I hate this smell", "This is revolting",
                "This is awful", "I can't stand this", "This is terrible",
                "This disgusts me"
            ],
            'Fear': [
                "I'm scared", "This frightens me", "I'm afraid", "This worries me",
                "I'm nervous", "This makes me anxious", "I'm concerned about this",
                "This scares me", "I feel worried", "This makes me uncomfortable"
            ],
            'Happy': [
                "I'm happy", "This makes me feel good", "I love this", "This is great",
                "I'm excited", "This is wonderful", "I feel good about this",
                "This makes me smile", "I'm pleased with this", "This brings me joy"
            ],
            'Neutral': [
                "This is fine", "I understand", "That's normal", "I see",
                "This is okay", "I'm listening", "That makes sense", "I agree",
                "This is usual", "I know"
            ],
            'Sad': [
                "I'm sad", "This makes me feel bad", "I'm disappointed", "This hurts",
                "I feel down", "This is depressing", "I'm upset", "This makes me cry",
                "I feel terrible", "This is heartbreaking"
            ]
        }

        # Complex templates for diversity
        self.complex_templates = {
            'Anger': [
                "I am absolutely furious about {situation}", "This makes me so angry I could {action}",
                "I hate it when {event} happens like this", "I'm livid about {problem} right now",
                "This is infuriating and {feeling}",
            ],
            'Disgust': [
                "This is absolutely disgusting and {feeling}", "I find {thing} completely repulsive",
                "That {object} makes me sick to my stomach", "How revolting and {adjective} this is",
                "I'm nauseated by {situation}",
            ],
            'Fear': [
                "I'm terrified of {situation} happening", "This scares me more than {comparison}",
                "I'm afraid that {event} will occur", "I feel anxious about {situation}",
                "This frightens me {intensifier}",
            ],
            'Happy': [
                "I'm so happy about {situation}", "This brings me {feeling} and joy",
                "I feel wonderful about {event}", "I'm excited that {situation} happened",
                "This makes me smile {intensifier}",
            ],
            'Neutral': [
                "I need to {action} the {object}", "The {event} is scheduled for {time}",
                "Please {action} the {document}", "The {meeting} will be held",
                "I have to {task} before {time}",
            ],
            'Sad': [
                "I feel so sad about {situation}", "This makes me {feeling} and depressed",
                "I'm heartbroken about {loss}", "I feel down because of {problem}",
                "This brings tears to my eyes",
            ]
        }

        # Variables for templates
        self.variables = {
            'situation': ['work', 'home', 'this', 'that', 'everything'],
            'action': ['scream', 'leave', 'stop', 'fix', 'change'],
            'event': ['problems', 'issues', 'things', 'stuff', 'situations'],
            'problem': ['this mess', 'the situation', 'what happened', 'this issue'],
            'feeling': ['wrong', 'bad', 'terrible', 'awful', 'horrible'],
            'thing': ['this', 'that', 'it', 'everything'],
            'object': ['thing', 'situation', 'problem', 'issue'],
            'adjective': ['bad', 'terrible', 'awful', 'horrible'],
            'comparison': ['anything', 'everything', 'other things'],
            'intensifier': ['so much', 'a lot', 'completely'],
            'time': ['today', 'now', 'soon', 'later'],
            'task': ['do', 'finish', 'complete', 'handle'],
            'document': ['work', 'things', 'stuff'],
            'meeting': ['thing', 'event', 'session'],
            'loss': ['this', 'what happened', 'the situation']
        }

    def fill_template(self, template: str) -> str:
        """Fill template with variables"""
        text = template
        for var_type, options in self.variables.items():
            placeholder = f"{{{var_type}}}"
            if placeholder in text:
                text = text.replace(placeholder, random.choice(options))

        import re
        text = re.sub(r'\{[^}]+\}', 'something', text)
        return text

    def generate_enhanced_synthetic(self) -> List[Dict]:
        """Generate enhanced synthetic data"""
        print(f"Generating enhanced synthetic data ({self.config.synthetic_samples_per_emotion} per emotion)...")

        synthetic_data = []
        for emotion in self.config.emotion_classes:
            audio_templates = self.audio_like_templates[emotion]
            complex_templates = self.complex_templates.get(emotion, audio_templates)

            for i in range(self.config.synthetic_samples_per_emotion):
                if i < self.config.synthetic_samples_per_emotion * 0.6:
                    text = random.choice(audio_templates)
                else:
                    template = random.choice(complex_templates)
                    text = self.fill_template(template)

                synthetic_data.append({
                    'text': text, 'emotion': emotion, 'dataset': 'synthetic',
                    'data_type': 'synthetic', 'actor_id': f"synthetic_{i}",
                    'audio_file': None, 'id': str(uuid.uuid4())
                })

        print(f"Generated {len(synthetic_data)} enhanced synthetic samples")
        return synthetic_data

class EnhancedAudioProcessor:
    """Enhanced audio processing with better transcription and augmentation"""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.crema_path = Path(config.crema_dataset_path)
        self.ravdess_path = Path(config.ravdess_dataset_path)

        self.crema_emotions = {'ANG': 'Anger', 'DIS': 'Disgust', 'FEA': 'Fear', 'HAP': 'Happy', 'NEU': 'Neutral', 'SAD': 'Sad'}
        self.ravdess_emotions = {1: 'Neutral', 2: 'Neutral', 3: 'Happy', 4: 'Sad', 5: 'Anger', 6: 'Fear', 7: 'Disgust', 8: 'Neutral'}

        if WHISPER_AVAILABLE and whisper is not None:
            try:
                print(f"Loading enhanced Whisper model ({config.whisper_model_size})...")
                self.whisper_model = whisper.load_model(config.whisper_model_size)
                print("Enhanced Whisper model loaded")
            except Exception as e:
                print(f"Could not load enhanced Whisper, falling back to base: {e}")
                self.whisper_model = whisper.load_model("base") if WHISPER_AVAILABLE else None
        else:
            self.whisper_model = None

        self.fallback_templates = {
            'Anger': ["I'm angry", "I hate this", "This makes me mad"],
            'Disgust': ["This is gross", "This is disgusting", "This is nasty"],
            'Fear': ["I'm scared", "I'm afraid", "This worries me"],
            'Happy': ["I'm happy", "I love this", "This is great"],
            'Neutral': ["This is fine", "I understand", "That's okay"],
            'Sad': ["I'm sad", "This hurts", "I'm disappointed"]
        }

    def preprocess_audio(self, audio_path: str, is_training: bool = False):
        """Enhanced audio preprocessing with optional augmentation"""
        try:
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, duration=self.config.max_audio_length)

                # Augmentation for training data
                if is_training and self.config.use_audio_augmentation:
                    # Add noise
                    if random.random() < 0.5:
                        noise = np.random.randn(len(audio))
                        audio = audio + self.config.audio_noise_factor * noise
                    # Pitch shift
                    if random.random() < 0.5:
                        n_steps = random.randint(-self.config.audio_pitch_shift_steps, self.config.audio_pitch_shift_steps)
                        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

                audio = librosa.effects.preemphasis(audio)
                audio = librosa.util.normalize(audio)
                return audio, sr
            else: # Fallback
                audio, sr = soundfile.read(audio_path)
                return audio, sr
        except Exception as e:
            print(f"Audio preprocessing failed for {audio_path}: {e}")
            return None, None

    def enhanced_transcribe(self, audio_path: str, emotion: str, is_training: bool = False) -> str:
        """Enhanced transcription with multiple attempts"""
        if not self.whisper_model: return random.choice(self.fallback_templates[emotion])

        try:
            audio, sr = self.preprocess_audio(audio_path, is_training=is_training)
            if audio is None: return random.choice(self.fallback_templates[emotion])

            best_text, best_confidence = "", 0.0
            attempts = [{"temperature": 0.0, "language": "en"}, {"temperature": 0.2, "language": "en"}, {"temperature": 0.0}]

            for params in attempts:
                try:
                    result = self.whisper_model.transcribe(audio, **params)
                    text = result["text"].strip()
                    confidence = self.estimate_transcription_quality(text, emotion)
                    if confidence > best_confidence:
                        best_confidence, best_text = confidence, text
                    if confidence > self.config.audio_quality_threshold: break
                except Exception: continue

            return best_text if len(best_text) >= self.config.min_text_length and best_confidence > 0.3 else random.choice(self.fallback_templates[emotion])
        except Exception: return random.choice(self.fallback_templates[emotion])

    def estimate_transcription_quality(self, text: str, emotion: str) -> float:
        """Estimate transcription quality"""
        if not text or len(text) < 2: return 0.0
        quality = 0.5
        if 3 <= len(text) <= 50: quality += 0.2
        words = text.split()
        if 2 <= len(words) <= 15: quality += 0.1
        emotion_words = {'Anger': ['angry', 'mad', 'hate'], 'Disgust': ['disgusting', 'gross', 'nasty'], 'Fear': ['scared', 'afraid', 'nervous'], 'Happy': ['happy', 'good', 'great'], 'Neutral': ['okay', 'fine', 'normal'], 'Sad': ['sad', 'hurt', 'disappointed']}
        if any(word in text.lower() for word in emotion_words.get(emotion, [])): quality += 0.1
        return min(quality, 1.0)

    def download_datasets(self):
        """Download datasets"""
        if not self.crema_path.exists() and KAGGLEHUB_AVAILABLE:
            try:
                print("Downloading CREMA-D...")
                path = kagglehub.dataset_download("ejlok1/cremad")
                self.crema_path.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in tqdm(list(Path(path).glob("**/*.wav")), desc="Copying CREMA-D"):
                    shutil.copy2(f, self.crema_path / f.name)
            except Exception as e: print(f"CREMA-D download failed: {e}")

        if not self.ravdess_path.exists() and KAGGLEHUB_AVAILABLE:
            try:
                print("Downloading RAVDESS...")
                path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
                self.ravdess_path.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in tqdm(list(Path(path).glob("**/*.wav")), desc="Copying RAVDESS"):
                    shutil.copy2(f, self.ravdess_path / f.name)
            except Exception as e: print(f"RAVDESS download failed: {e}")

    def parse_crema_filename(self, filename: str) -> Optional[Dict]:
        try:
            parts = Path(filename).stem.split('_')
            return {'actor_id': parts[0], 'emotion': self.crema_emotions.get(parts[2]), 'dataset': 'crema_d'} if len(parts) >= 3 else None
        except: return None

    def parse_ravdess_filename(self, filename: str) -> Optional[Dict]:
        try:
            parts = Path(filename).stem.split('-')
            return {'actor_id': parts[6], 'emotion': self.ravdess_emotions.get(int(parts[2])), 'dataset': 'ravdess'} if len(parts) >= 7 else None
        except: return None

    def process_enhanced_audio(self) -> List[Dict]:
        """Process audio with enhanced methods"""
        print("Processing audio with enhanced methods...")
        self.download_datasets()
        audio_data = []

        # Process datasets
        datasets = {
            "CREMA-D": (list(self.crema_path.glob("*.wav")) if self.crema_path.exists() else [], self.parse_crema_filename),
            "RAVDESS": (list(self.ravdess_path.glob("*.wav")) if self.ravdess_path.exists() else [], self.parse_ravdess_filename)
        }

        for name, (files, parser) in datasets.items():
            print(f"Processing {len(files)} {name} files with enhanced transcription...")
            for audio_file in tqdm(files, desc=f"Enhanced {name}"):
                try:
                    metadata = parser(audio_file.name)
                    if metadata and metadata['emotion']:
                        # Augmentations are applied here for training data
                        text = self.enhanced_transcribe(str(audio_file), metadata['emotion'], is_training=True)
                        audio_data.append({
                            'text': text, 'emotion': metadata['emotion'], 'dataset': metadata['dataset'],
                            'data_type': 'audio', 'actor_id': metadata['actor_id'],
                            'audio_file': str(audio_file), 'id': str(uuid.uuid4())
                        })
                except Exception: continue

        print(f"Processed {len(audio_data)} enhanced audio samples")
        return audio_data

class EnhancedDataProcessor:
    """Enhanced data processor with improved splits"""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.synthetic_generator = EnhancedSyntheticGenerator(config)
        self.audio_processor = EnhancedAudioProcessor(config)

    def create_optimized_splits(self, synthetic_data: List[Dict], audio_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create optimized splits for 40% target"""
        print("Creating optimized data splits (80% audio for training)...")
        audio_by_emotion = defaultdict(list)
        for item in audio_data: audio_by_emotion[item['emotion']].append(item)

        audio_train, audio_test, audio_val = [], [], []
        for emotion, items in audio_by_emotion.items():
            random.shuffle(items)
            n_test = max(1, int(len(items) * self.config.audio_test_ratio))
            n_val = max(1, int(len(items) * self.config.audio_val_ratio))
            audio_test.extend(items[:n_test])
            audio_val.extend(items[n_test:n_test + n_val])
            audio_train.extend(items[n_test + n_val:])

        train_data = synthetic_data + audio_train
        random.shuffle(train_data)
        random.shuffle(audio_test) # Corrected variable name
        random.shuffle(audio_val)  # Corrected variable name
        return train_data, audio_test, audio_val

    def process_datasets(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process datasets with enhancements"""
        print("ENHANCED HYBRID DATA PROCESSING")
        synthetic_data = self.synthetic_generator.generate_enhanced_synthetic()
        audio_data = self.audio_processor.process_enhanced_audio()

        if not audio_data:
            print("ERROR: No audio data available"); return [], [], []

        emotion_counts = Counter(item['emotion'] for item in (synthetic_data + audio_data))
        print(f"\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            syn_count = sum(1 for item in synthetic_data if item['emotion'] == emotion)
            aud_count = sum(1 for item in audio_data if item['emotion'] == emotion)
            print(f"   {emotion}: {count} (syn: {syn_count}, audio: {aud_count})")

        return self.create_optimized_splits(synthetic_data, audio_data)

class EnhancedModel(nn.Module):
    """Enhanced model architecture for 40% target performance"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"Loading enhanced {config.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.backbone = AutoModel.from_pretrained(config.base_model)

        if hasattr(self.backbone, 'encoder'):
            for layer in self.backbone.encoder.layer[:6]: # Freeze more layers
                for param in layer.parameters(): param.requires_grad = False

        if config.use_multi_layer_pooling:
            self.layer_weights = nn.Parameter(torch.ones(4))
        if config.use_attention_pooling:
            self.attention = nn.MultiheadAttention(config.hidden_size, 8, 0.1, batch_first=True)
            self.attention_norm = nn.LayerNorm(config.hidden_size)

        layers, input_size = [], config.hidden_size
        for hidden_dim in config.hidden_layers:
            layers.extend([nn.Linear(input_size, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(config.dropout_rate)])
            input_size = hidden_dim
        layers.append(nn.Linear(input_size, len(config.emotion_classes)))
        self.classifier = nn.Sequential(*layers)
        self._init_weights()
        print(f"Enhanced model ready with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters")

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
             outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=self.config.use_multi_layer_pooling)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.config.use_multi_layer_pooling)

        if self.config.use_multi_layer_pooling:
            hidden_states = outputs.hidden_states[-4:]
            layer_weights = F.softmax(self.layer_weights, dim=0)
            sequence_output = sum(weight * state for weight, state in zip(layer_weights, hidden_states))
        else:
            sequence_output = outputs.last_hidden_state

        if self.config.use_attention_pooling:
            attn_out, _ = self.attention(sequence_output, sequence_output, sequence_output, key_padding_mask=~attention_mask.bool())
            attn_out = self.attention_norm(attn_out + sequence_output)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = sequence_output[:, 0, :]

        logits = self.classifier(pooled)
        return {'logits': logits, 'probabilities': F.softmax(logits, dim=-1)}

class EnhancedDataset(Dataset):
    """Enhanced dataset with mixup augmentation"""

    def __init__(self, data, tokenizer, config, is_training=False):
        self.data, self.tokenizer, self.config, self.is_training = data, tokenizer, config, is_training
        self.emotion_to_id = {e: i for i, e in enumerate(config.emotion_classes)}

        counts = Counter(item['emotion'] for item in data)
        self.class_weights = {e: 1.0 / counts.get(e, 1) for e in config.emotion_classes}

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = ' '.join(item['text'].strip().split()) or f"sample from {item['emotion'].lower()}"
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.config.max_sequence_length, return_tensors='pt')

        result = {
            'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.emotion_to_id[item['emotion']], dtype=torch.long)
        }

        if self.is_training:
            weight = self.class_weights.get(item['emotion'], 1.0) * (self.config.audio_weight if item['data_type'] == 'audio' else 1.0)
            result['sample_weight'] = torch.tensor(weight, dtype=torch.float)

            result.update({
                'mixup_input_ids': result['input_ids'].clone(), 'mixup_attention_mask': result['attention_mask'].clone(),
                'mixup_labels': result['labels'].clone(), 'mixup_lambda': torch.tensor(1.0, dtype=torch.float)
            })

            if self.config.use_mixup and random.random() < 0.3:
                other_idx = random.randint(0, len(self.data) - 1)
                other_item = self.data[other_idx]
                if other_item['emotion'] != item['emotion']:
                    other_enc = self.tokenizer(other_item['text'].strip(), truncation=True, padding='max_length', max_length=self.config.max_sequence_length, return_tensors='pt')
                    result.update({
                        'mixup_input_ids': other_enc['input_ids'].squeeze(),
                        'mixup_attention_mask': other_enc['attention_mask'].squeeze(),
                        'mixup_labels': torch.tensor(self.emotion_to_id[other_item['emotion']], dtype=torch.long),
                        'mixup_lambda': torch.tensor(np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha), dtype=torch.float)
                    })
        return result

class FGM:
    """Fast Gradient Method for Adversarial Training"""
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad: param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: param.data = self.backup[name]
        self.backup = {}

class EnhancedTrainer:
    """Enhanced trainer with all optimization strategies"""

    def __init__(self, config):
        self.config = config
        self.model = EnhancedModel(config).to(config.device)
        self.fgm = FGM(self.model) if config.use_adversarial_training else None

        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'backbone' in n], 'lr': config.learning_rate * 0.1},
            {'params': [p for n, p in self.model.named_parameters() if 'backbone' not in n], 'lr': config.learning_rate}
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-6)

        if config.use_label_smoothing:
            self.criterion = LabelSmoothingLoss(len(config.emotion_classes), smoothing=config.label_smoothing_factor, reduction='none')
        elif config.use_focal_loss:
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.ema = EMAModel(self.model, decay=config.ema_decay) if config.use_ema else None
        self.best_val_accuracy, self.best_test_accuracy = 0, 0
        self.patience, self.max_patience = 0, 10
        self.training_history = []

    def mixup_criterion(self, pred, y_a, y_b, lam):
        lam = lam.view(-1)
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc="Enhanced Training"):
            input_ids, attention_mask, labels, sample_weights = [b.to(self.config.device) for b in batch.values() if isinstance(b, torch.Tensor)][:4]
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)

            # Mixup loss calculation
            if self.config.use_mixup and 'mixup_lambda' in batch:
                 mixup_lambda, mixup_labels = batch['mixup_lambda'].to(self.config.device), batch['mixup_labels'].to(self.config.device)
                 if torch.any(mixup_lambda < 1.0):
                     loss = self.mixup_criterion(outputs['logits'], labels, mixup_labels, mixup_lambda)

            weighted_loss = (loss * sample_weights).mean()
            weighted_loss.backward()

            # Adversarial training step
            if self.fgm:
                self.fgm.attack()
                outputs_adv = self.model(input_ids, attention_mask)
                loss_adv = self.criterion(outputs_adv['logits'], labels)
                if self.config.use_mixup and 'mixup_lambda' in batch and torch.any(mixup_lambda < 1.0):
                    loss_adv = self.mixup_criterion(outputs_adv['logits'], labels, mixup_labels, mixup_lambda)

                weighted_loss_adv = (loss_adv * sample_weights).mean()
                weighted_loss_adv.backward()
                self.fgm.restore()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            if self.ema: self.ema.update()
            if hasattr(self, 'scheduler'): self.scheduler.step()

            total_loss += weighted_loss.item()
            preds = torch.argmax(outputs['logits'], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), correct / total

    def evaluate(self, data_loader, name=""):
        if self.ema: self.ema.apply_shadow()
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
                input_ids, attention_mask, labels = [b.to(self.config.device) for b in batch.values() if isinstance(b, torch.Tensor)]
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels).mean()

                total_loss += loss.item()
                preds = torch.argmax(outputs['logits'], dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if self.ema: self.ema.restore()
        return total_loss / len(data_loader), correct / total, all_preds, all_labels

    def train(self, train_data, test_data, val_data):
        print("STARTING ENHANCED TRAINING FOR 40% TARGET")
        train_dataset = EnhancedDataset(train_data, self.model.tokenizer, self.config, is_training=True)
        test_dataset = EnhancedDataset(test_data, self.model.tokenizer, self.config)
        val_dataset = EnhancedDataset(val_data, self.model.tokenizer, self.config)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader, "Validation")

            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                _, self.best_test_accuracy, _, _ = self.evaluate(test_loader, "Test (Best Model)")
                self.patience = 0
                self.save_model("best_enhanced_model.pt")
                print(f"   NEW BEST! Val Acc: {val_acc:.4f}, Test Acc: {self.best_test_accuracy:.4f}")
                print("\nDetailed Classification Report (Best Model on Validation):")
                print(classification_report(val_labels, val_preds, target_names=self.config.emotion_classes, digits=4, zero_division=0))
            else:
                self.patience += 1

            if self.patience >= self.max_patience:
                print(f"Early stopping at epoch {epoch + 1}"); break

        print(f"\nENHANCED TRAINING COMPLETE!")
        print(f"Best Val Acc: {self.best_val_accuracy:.4f}, Best Test Acc: {self.best_test_accuracy:.4f}")
        return {'best_val_accuracy': self.best_val_accuracy, 'best_test_accuracy': self.best_test_accuracy}

    def save_model(self, path):
        save_data = {'model_state_dict': self.model.state_dict(), 'config': self.config}
        if self.ema: save_data['ema_state_dict'] = self.ema.shadow
        torch.save(save_data, path)

def run_enhanced_training():
    """Run enhanced training for 40% target"""
    print("ENHANCED HYBRID TRAINING FOR 40% TARGET")
    print("=" * 60)
    print("NEW Strategies: roberta-large, Adversarial Training, Label Smoothing, Audio Augmentation")

    try:
        config = EnhancedConfig()
        processor = EnhancedDataProcessor(config)
        train_data, test_data, val_data = processor.process_datasets()

        if not train_data: return None

        trainer = EnhancedTrainer(config)
        results = trainer.train(train_data, test_data, val_data)

        print("\nFINAL ENHANCED RESULTS:")
        print(f"Best Audio Validation: {results['best_val_accuracy']:.4f}")
        print(f"Best Audio Test: {results['best_test_accuracy']:.4f}")

    except Exception as e:
        print(f"Enhanced training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_enhanced_training()
