# =============================================================================
# AI-Powered Strategic Clarity Platform ("Clarity Engine") 
# BERT/DistilBERT FINE-TUNING
# Jessica Cervi's 4/30/2025 Core Focus: Fine-tuned NLP Classifier Implementation
# Advanced Transformer-based Classification for Organizational Pattern Detection
# =============================================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Transformer imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        BertTokenizer, BertForSequenceClassification,
        TrainingArguments, Trainer, pipeline,
        EarlyStoppingCallback, IntervalStrategy
    )
    from transformers.trainer_utils import EvalPrediction
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers library not found")
    print("Install with: pip install transformers torch")

# Check for GPU availability
if TRANSFORMERS_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "="*70)
print("BERT/DistilBERT FINE-TUNING FOR WORKPLACE POWER DYNAMICS")
print("Jessica Cervi's 4/30/2025 Core Focus Implementation")
print("="*70)

# =============================================================================
# 1. CUSTOM DATASET CLASS FOR WORKPLACE NARRATIVES
# =============================================================================

class WorkplacePowerDataset(Dataset):
    """
    Custom PyTorch Dataset for workplace power dynamics classification
    Optimized for BERT/DistilBERT fine-tuning with workplace-specific preprocessing
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset for workplace narrative classification
        
        Args:
            texts: List of workplace narrative texts
            labels: List of corresponding power dynamic pattern labels
            tokenizer: HuggingFace tokenizer (BERT/DistilBERT)
            max_length: Maximum sequence length for tokenization
        """
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize workplace narrative
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# 2. ADVANCED BERT CONFIGURATION FOR WORKPLACE DOMAIN
# =============================================================================

class WorkplaceBERTClassifier:
    """
    Advanced BERT classifier specifically designed for workplace power dynamics
    Implements Jessica's requirements for fine-tuned organizational pattern detection
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=None):
        """
        Initialize BERT classifier for workplace patterns
        
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of workplace power dynamic patterns
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, df, text_column='narrative_text', label_column='pattern', test_size=0.2):
        """
        Prepare workplace narrative data for BERT fine-tuning
        """
        print(f"\n📊 Preparing workplace data for BERT fine-tuning...")
        print(f"Total narratives: {len(df)}")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(df[label_column])
        self.num_labels = len(self.label_encoder.classes_)
        
        print(f"Workplace patterns identified: {self.num_labels}")
        for i, pattern in enumerate(self.label_encoder.classes_):
            count = sum(df[label_column] == pattern)
            print(f"  {i}: {pattern} ({count} examples)")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_column].values,
            labels_encoded,
            test_size=test_size,
            random_state=42,
            stratify=labels_encoded
        )
        
        print(f"\nData split:")
        print(f"  Training examples: {len(X_train)}")
        print(f"  Test examples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_model(self):
        """
        Initialize BERT model and tokenizer for workplace classification
        """
        print(f"\n🤖 Initializing {self.model_name} for workplace power dynamics...")
        
        if 'distilbert' in self.model_name.lower():
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
        
        self.model.to(device)
        
        print(f"✅ Model initialized with {self.num_labels} workplace pattern classes")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_datasets(self, X_train, X_test, y_train,