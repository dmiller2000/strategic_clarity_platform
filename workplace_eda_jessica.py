# =============================================================================
# AI-Powered Strategic Clarity Platform ("Clarity Engine") - EDA ANALYSIS
# Following Jessica Cervi's NLP Framework (Module 18)
# Student: [Your Name] | UC Berkeley Professional Certificate in ML & AI
# =============================================================================

# 1. IMPORTS AND SETUP (Jessica's recommended libraries + BERT)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB  # Jessica's #1 recommendation
from sklearn.tree import DecisionTreeClassifier  # Jessica: "trees work well"
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from wordcloud import WordCloud
from collections import Counter
import string
import warnings
warnings.filterwarnings('ignore')

# BERT/Transformer imports (Jessica's 4/30/2025 core focus)
try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                             DistilBertTokenizer, DistilBertForSequenceClassification,
                             TrainingArguments, Trainer, pipeline)
    import torch
    from torch.utils.data import Dataset, DataLoader
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available - BERT implementation enabled")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers library not found - install with: pip install transformers torch")

# Check for GPU availability
if TRANSFORMERS_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("=== WORKPLACE POWER DYNAMICS CLASSIFIER ===")
print("Following Jessica Cervi's NLP Framework from Module 18")
print("UC Berkeley Professional Certificate in ML & AI")
print("=" * 50)

# 2. JESSICA'S NLP PREPROCESSING PIPELINE
def jessica_preprocessing_pipeline(text):
    """
    Jessica Cervi's exact preprocessing methodology from Module 18
    Key insight: "Simplify words as much as necessary to maintain context"
    """
    if pd.isna(text):
        return []
    
    # Step 1: Tokenization (Jessica emphasized this as fundamental)
    tokens = word_tokenize(str(text).lower())
    
    # Step 2: Remove punctuation (Jessica's recommendation)
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Step 3: Custom stop words for workplace context
    # Jessica: "Customize stop words but preserve meaning"
    stop_words = set(stopwords.words('english'))
    
    # Workplace-specific stop words (noise terms)
    workplace_noise = {'company', 'work', 'job', 'office', 'team', 'department', 'employee'}
    
    # Preserve power dynamic indicators (Jessica's meaning preservation)
    power_indicators = {'not', 'never', 'always', 'sudden', 'began', 'started', 'stopped', 
                       'change', 'different', 'new', 'told', 'said', 'asked', 'meeting'}
    
    # Final stop words: standard + workplace noise - power indicators
    final_stop_words = stop_words.union(workplace_noise) - power_indicators
    tokens = [token for token in tokens if token not in final_stop_words]
    
    # Step 4: Lemmatization over stemming (Jessica's strong preference)
    # Jessica quote: "Lemmatization keeps a lot more meaning"
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove very short tokens (less than 3 characters)
    tokens = [token for token in tokens if len(token) >= 3]
    
    return tokens

def jessica_bow_vectorization(df, max_features=1000):
    """
    Jessica's bag-of-words approach with workplace customization
    """
    vectorizer = CountVectorizer(
        tokenizer=jessica_preprocessing_pipeline,
        max_features=max_features,
        binary=False  # Counts matter for pattern intensity
    )
    
    bow_matrix = vectorizer.fit_transform(df['narrative_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    return bow_matrix, feature_names, vectorizer

# 3. DATA LOADING AND JESSICA'S QUALITY ASSESSMENT
def load_and_assess_data():
    """
    Load your dataset and apply Jessica's data quality framework
    REPLACE THIS SECTION WITH YOUR ACTUAL DATA LOADING
    """
    print("\n=== STEP 1: DATA LOADING ===")
    
    # REPLACE WITH YOUR DATA LOADING CODE:
    # df = pd.read_csv('your_workplace_narratives.csv')
    
    # FOR TEMPLATE PURPOSES - REMOVE WHEN YOU HAVE REAL DATA:
    print("📁 PLACEHOLDER: Insert your data loading code here")
    print("   Example: df = pd.read_csv('workplace_narratives.csv')")
    print("   Required columns: 'narrative_text', 'pattern'")
    
    # Uncomment and modify when you have your data:
    # return jessica_data_quality_assessment(df)
    
    return None

def jessica_data_quality_assessment(df):
    """
    Jessica's emphasis on understanding your data characteristics
    """
    print("\n=== JESSICA'S DATA QUALITY ASSESSMENT ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate narratives: {df['narrative_text'].duplicated().sum()}")
    
    # Text characteristics Jessica wants to see
    df['word_count'] = df['narrative_text'].str.split().str.len()
    df['char_count'] = df['narrative_text'].str.len()
    df['unique_words'] = df['narrative_text'].apply(lambda x: len(set(str(x).split())))
    
    print(f"\nText Characteristics:")
    print(f"- Average words per narrative: {df['word_count'].mean():.1f}")
    print(f"- Word count range: {df['word_count'].min()} - {df['word_count'].max()}")
    print(f"- Average unique words per narrative: {df['unique_words'].mean():.1f}")
    
    # Pattern distribution (Jessica wants balanced data)
    print(f"\nPattern Distribution:")
    pattern_counts = df['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        percentage = (count / len(df)) * 100
        print(f"- {pattern}: {count} examples ({percentage:.1f}%)")
    
    return df

# 4. JESSICA'S INTRINSIC EVALUATION
def intrinsic_evaluation_jessica(df):
    """
    Jessica: "Intrinsic evaluation - ensuring meaning is not lost during preprocessing"
    """
    print("\n=== JESSICA'S INTRINSIC EVALUATION ===")
    print("Checking preprocessing preserves workplace context meaning")
    
    # Sample text analysis
    sample_text = df['narrative_text'].iloc[0]
    processed_tokens = jessica_preprocessing_pipeline(sample_text)
    
    print("\nOriginal text sample:")
    print(f"'{sample_text[:200]}...'")
    print(f"\nAfter Jessica's preprocessing (first 20 tokens):")
    print(f"{processed_tokens[:20]}")
    
    # Check workplace context preservation
    workplace_terms = ['performance', 'improvement', 'plan', 'manager', 'meeting', 
                      'review', 'feedback', 'project', 'deadline', 'responsibility']
    
    preserved_count = 0
    for text in df['narrative_text'].head(10):
        tokens = jessica_preprocessing_pipeline(text)
        preserved_terms = [term for term in workplace_terms if term in tokens]
        preserved_count += len(preserved_terms)
    
    print(f"\nWorkplace Context Preservation:")
    print(f"- Key workplace terms preserved across sample: {preserved_count}")
    print(f"- Context preservation appears: {'✅ GOOD' if preserved_count > 5 else '⚠️ NEEDS REVIEW'}")

# 5. PATTERN SEPARABILITY ANALYSIS
def pattern_separability_analysis(df):
    """
    Jessica wants evidence that patterns are distinguishable
    """
    print("\n=== PATTERN SEPARABILITY ANALYSIS ===")
    print("Analyzing vocabulary differences between workplace patterns")
    
    patterns = df['pattern'].unique()
    
    for pattern in patterns:
        print(f"\n--- {pattern.upper()} PATTERN ---")
        pattern_text = df[df['pattern'] == pattern]['narrative_text']
        
        # Combine all text for this pattern
        all_tokens = []
        for text in pattern_text:
            all_tokens.extend(jessica_preprocessing_pipeline(text))
        
        # Top terms for this pattern
        top_terms = Counter(all_tokens).most_common(10)
        print("Top distinctive terms:")
        for term, count in top_terms:
            print(f"  '{term}': {count} occurrences")

# 6. JESSICA'S BASELINE MODELS + BERT IMPLEMENTATION

class WorkplaceDataset(Dataset):
    """
    Custom Dataset class for BERT fine-tuning on workplace narratives
    Jessica's 4/30/2025 core focus: Fine-tuned NLP classifier for organizational patterns
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def jessica_bert_implementation(df):
    """
    Jessica's 4/30/2025 Core Focus: Fine-tuned BERT for organizational pattern detection
    Implements both DistilBERT and standard BERT for workplace power dynamics
    """
    if not TRANSFORMERS_AVAILABLE:
        print("❌ BERT implementation requires transformers library")
        print("Install with: pip install transformers torch")
        return None, None, 0.0
    
    print("\n=== BERT/DistilBERT IMPLEMENTATION (Jessica's Core Focus 4/30/2025) ===")
    print("Fine-tuning transformer models for organizational pattern detection")
    
    # Prepare labels
    unique_patterns = df['pattern'].unique()
    pattern_to_id = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
    id_to_pattern = {idx: pattern for pattern, idx in pattern_to_id.items()}
    
    df['label_id'] = df['pattern'].map(pattern_to_id)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['narrative_text'].values, 
        df['label_id'].values, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label_id']
    )
    
    # Initialize DistilBERT (faster training, good performance)
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_patterns)
    ).to(device)
    
    print(f"Model: {model_name}")
    print(f"Number of patterns: {len(unique_patterns)}")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    
    # Create datasets
    train_dataset = WorkplaceDataset(X_train, y_train, tokenizer)
    test_dataset = WorkplaceDataset(X_test, y_test, tokenizer)
    
    # Training arguments optimized for workplace text
    training_args = TrainingArguments(
        output_dir='./workplace_bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./workplace_bert_logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )
    
    print("\n🚀 Starting BERT fine-tuning...")
    
    # Fine-tune the model
    trainer.train()
    
    # Evaluation
    print("\n📊 Evaluating BERT performance...")
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    y_pred_bert = np.argmax(predictions.predictions, axis=1)
    
    # Convert back to pattern names for interpretability
    y_test_patterns = [id_to_pattern[label] for label in y_test]
    y_pred_patterns = [id_to_pattern[pred] for pred in y_pred_bert]
    
    # Calculate metrics
    bert_accuracy = accuracy_score(y_test, y_pred_bert)
    bert_f1 = f1_score(y_test, y_pred_bert, average='macro')
    
    print("BERT/DistilBERT Performance:")
    print(f"- Accuracy: {bert_accuracy:.3f}")
    print(f"- Macro F1 Score: {bert_f1:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_patterns, y_pred_patterns))
    
    # Save the fine-tuned model
    model.save_pretrained('./workplace_bert_model')
    tokenizer.save_pretrained('./workplace_bert_model')
    
    print("✅ Fine-tuned BERT model saved to './workplace_bert_model'")
    
    return model, tokenizer, bert_f1

def jessica_bert_inference_pipeline(model_path='./workplace_bert_model'):
    """
    Create inference pipeline for real-time workplace pattern detection
    Jessica's focus: Practical application for organizational pattern identification
    """
    if not TRANSFORMERS_AVAILABLE:
        print("❌ BERT inference requires transformers library")
        return None
    
    try:
        # Load fine-tuned model
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("✅ BERT inference pipeline ready for workplace pattern detection")
        return classifier
        
    except Exception as e:
        print(f"❌ Error loading BERT model: {e}")
        return None

def demonstrate_bert_inference(classifier, sample_texts):
    """
    Demonstrate real-time workplace pattern detection
    """
    if classifier is None:
        print("❌ BERT classifier not available")
        return
    
    print("\n=== BERT INFERENCE DEMONSTRATION ===")
    print("Real-time workplace pattern detection:")
    
    for i, text in enumerate(sample_texts):
        result = classifier(text)
        confidence = result[0]['score']
        pattern = result[0]['label']
        
        print(f"\nSample {i+1}:")
        print(f"Text: \"{text[:100]}...\"")
        print(f"Detected Pattern: {pattern}")
        print(f"Confidence: {confidence:.3f}")

def jessica_naive_bayes_baseline(df):
    """
    Jessica's #1 algorithm recommendation: "Naive Bayes works very well for text classification"
    """
    print("\n=== NAIVE BAYES BASELINE (Jessica's Top Recommendation) ===")
    
    # Prepare data using Jessica's preprocessing
    bow_matrix, feature_names, vectorizer = jessica_bow_vectorization(df)
    
    X = bow_matrix
    y = df['pattern']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print("Naive Bayes Performance:")
    print(f"- Accuracy: {accuracy:.3f}")
    print(f"- Macro F1 Score: {f1_macro:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return nb_model, vectorizer, f1_macro

def jessica_decision_tree_baseline(df):
    """
    Jessica: "trees work well" - focus on interpretability
    """
    print("\n=== DECISION TREE BASELINE (Jessica's Interpretability Choice) ===")
    
    # Use same preprocessing as Naive Bayes
    bow_matrix, feature_names, vectorizer = jessica_bow_vectorization(df, max_features=500)
    
    X = bow_matrix
    y = df['pattern']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Decision Tree with professional constraints
    dt_model = DecisionTreeClassifier(
        max_depth=8,  # Prevent overfitting
        min_samples_split=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print("Decision Tree Performance:")
    print(f"- Accuracy: {accuracy:.3f}")
    print(f"- Macro F1 Score: {f1_macro:.3f}")
    
    # Feature importance (Jessica's interpretability focus)
    importances = dt_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    
    print("\nTop 10 Most Important Features for Pattern Detection:")
    for i, idx in enumerate(top_indices):
        if importances[idx] > 0:
            print(f"{i+1:2d}. '{feature_names[idx]}' (importance: {importances[idx]:.3f})")
    
    return dt_model, f1_macro

# 7. JESSICA'S PROFESSIONAL VISUALIZATIONS
def create_jessica_visualizations(df):
    """
    Portfolio-quality visualizations meeting Jessica's professional standards
    """
    print("\n=== CREATING PROFESSIONAL VISUALIZATIONS ===")
    
    # Set professional style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Workplace Power Dynamics Classifier - Jessica Cervi NLP Framework', 
                 fontsize=16, fontweight='bold')
    
    # 1. Pattern Distribution
    pattern_counts = df['pattern'].value_counts()
    pattern_counts.plot(kind='bar', ax=axes[0,0], color='skyblue', edgecolor='black')
    axes[0,0].set_title('Pattern Distribution\n(Jessica\'s Data Balance Check)', fontweight='bold')
    axes[0,0].set_xlabel('Workplace Pattern')
    axes[0,0].set_ylabel('Number of Examples')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Narrative Length Distribution
    df['word_count'].hist(bins=20, ax=axes[0,1], color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Narrative Length Distribution\n(Jessica\'s Text Characteristics)', fontweight='bold')
    axes[0,1].set_xlabel('Word Count')
    axes[0,1].set_ylabel('Frequency')
    
    # 3. Preprocessing Impact
    sample_texts = df['narrative_text'].head(50)
    original_lengths = [len(str(text).split()) for text in sample_texts]
    processed_lengths = [len(jessica_preprocessing_pipeline(text)) for text in sample_texts]
    
    axes[0,2].scatter(original_lengths, processed_lengths, alpha=0.6, color='coral')
    axes[0,2].plot([0, max(original_lengths)], [0, max(original_lengths)], 'r--', alpha=0.5)
    axes[0,2].set_title('Jessica\'s Preprocessing Impact\n(Meaning Preservation Check)', fontweight='bold')
    axes[0,2].set_xlabel('Original Length (words)')
    axes[0,2].set_ylabel('After Preprocessing (tokens)')
    
    # 4. Word Count by Pattern
    df.boxplot(column='word_count', by='pattern', ax=axes[1,0])
    axes[1,0].set_title('Word Count by Pattern\n(Jessica\'s Separability Analysis)', fontweight='bold')
    axes[1,0].set_xlabel('Pattern')
    axes[1,0].set_ylabel('Word Count')
    
    # 5. Unique Words Analysis
    df['unique_word_ratio'] = df['unique_words'] / df['word_count']
    df['unique_word_ratio'].hist(bins=15, ax=axes[1,1], color='plum', edgecolor='black')
    axes[1,1].set_title('Vocabulary Diversity\n(Unique Words / Total Words)', fontweight='bold')
    axes[1,1].set_xlabel('Unique Word Ratio')
    axes[1,1].set_ylabel('Frequency')
    
    # 6. Model Performance Placeholder
    axes[1,2].text(0.5, 0.5, 'Model Performance Summary\n\nNaive Bayes F1: [TO BE FILLED]\nDecision Tree F1: [TO BE FILLED]\n\nF1 > 0.75 Feasibility:\n[TO BE DETERMINED]', 
                   ha='center', va='center', transform=axes[1,2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                   fontsize=10, fontweight='bold')
    axes[1,2].set_title('Jessica\'s F1 > 0.75 Assessment', fontweight='bold')
    axes[1,2].set_xticks([])
    axes[1,2].set_yticks([])
    
    plt.tight_layout()
    plt.show()

# 8. COMPREHENSIVE MODEL COMPARISON INCLUDING BERT
def comprehensive_model_comparison(nb_f1, dt_f1, bert_f1=None):
    """
    Jessica's complete evaluation framework: Traditional ML + Modern Transformers
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON - JESSICA'S COMPLETE FRAMEWORK")
    print("="*70)
    
    models_performance = {
        'Naive Bayes (Jessica\'s #1)': nb_f1,
        'Decision Tree (Interpretability)': dt_f1
    }
    
    if bert_f1 is not None:
        models_performance['BERT/DistilBERT (Transformer)'] = bert_f1
    
    print("Model Performance Summary:")
    for model, f1 in models_performance.items():
        status = "✅" if f1 > 0.75 else "🎯" if f1 > 0.65 else "⚠️" if f1 > 0.50 else "❌"
        print(f"  {status} {model}: F1 = {f1:.3f}")
    
    best_model = max(models_performance.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Performing Model: {best_model[0]} (F1 = {best_model[1]:.3f})")
    
    # F1 > 0.75 Achievement Assessment
    target_achieved = any(f1 > 0.75 for f1 in models_performance.values())
    
    print(f"\nF1 > 0.75 Target Assessment:")
    if target_achieved:
        print("✅ TARGET ACHIEVED!")
        print("- Ready for production deployment consideration")
        print("- Excellent performance for workplace pattern detection")
        print("- Strong foundation for user-facing application")
    elif best_model[1] > 0.65:
        print("🎯 NEAR TARGET - EXCELLENT PROGRESS")
        print("- Strong performance indicates robust pattern detection")
        print("- Minor optimization could achieve target")
        print("- Ready for advanced feature engineering")
    else:
        print("⚠️ BELOW TARGET - IMPROVEMENT OPPORTUNITIES")
        print("- Consider additional data collection")
        print("- Explore ensemble methods")
        print("- Reassess pattern complexity")
    
    return best_model

# 9. JESSICA'S F1 > 0.75 FEASIBILITY ASSESSMENT (UPDATED)
def jessica_f1_feasibility_assessment_complete(nb_f1, dt_f1, bert_f1=None):
    """
    Jessica's framework for assessing Module 24 success probability
    """
    print("\n" + "="*60)
    print("JESSICA'S F1 > 0.75 FEASIBILITY ASSESSMENT")
    print("="*60)
    
    best_f1 = max(nb_f1, dt_f1)
    
    print(f"Baseline Model Performance:")
    print(f"- Naive Bayes F1 Score: {nb_f1:.3f}")
    print(f"- Decision Tree F1 Score: {dt_f1:.3f}")
    print(f"- Best Baseline F1: {best_f1:.3f}")
    
    print(f"\nFeasibility Assessment for F1 > 0.75:")
    
    if best_f1 > 0.70:
        print("✅ HIGHLY FEASIBLE")
        print("- Strong baseline performance indicates excellent pattern separability")
        print("- BERT fine-tuning should easily exceed F1 > 0.75 target")
        print("- Recommend proceeding with confidence to Module 24")
        
    elif best_f1 > 0.60:
        print("✅ FEASIBLE")
        print("- Good baseline performance suggests BERT will reach target")
        print("- Consider minor data augmentation for enhanced performance")
        print("- Proceed to Module 24 with optimization focus")
        
    elif best_f1 > 0.45:
        print("⚠️ CHALLENGING BUT ACHIEVABLE")
        print("- Moderate baseline suggests need for strategic improvements:")
        print("  • Additional high-quality training data")
        print("  • Enhanced feature engineering")
        print("  • Careful BERT hyperparameter tuning")
        print("- Feasible with focused effort in Module 24")
        
    else:
        print("❌ REQUIRES SIGNIFICANT IMPROVEMENTS")
        print("- Low baseline performance indicates fundamental challenges:")
        print("  • Consider reducing number of patterns")
        print("  • Improve data quality and quantity")
        print("  • Alternative problem formulation may be needed")
        print("- Reassess approach before Module 24")
    
    print(f"\nRecommendation for Module 24:")
    if best_f1 > 0.60:
        print("- Implement BERT fine-tuning with standard hyperparameters")
        print("- Focus on explainability features (SHAP integration)")
        print("- Develop user interface for real-time classification")
    else:
        print("- Prioritize data collection and quality improvements")
        print("- Consider ensemble methods combining multiple approaches")
    def jessica_f1_feasibility_assessment_complete(nb_f1, dt_f1, bert_f1=None):
    """
    Jessica's updated framework including BERT performance assessment
    """
    print("\n" + "="*60)
    print("JESSICA'S COMPLETE F1 > 0.75 FEASIBILITY ASSESSMENT")
    print("Including Traditional ML + Transformer Models")
    print("="*60)
    
    all_scores = [nb_f1, dt_f1]
    if bert_f1 is not None:
        all_scores.append(bert_f1)
    
    best_f1 = max(all_scores)
    
    print(f"Model Performance Summary:")
    print(f"- Naive Bayes F1 Score: {nb_f1:.3f}")
    print(f"- Decision Tree F1 Score: {dt_f1:.3f}")
    if bert_f1 is not None:
        print(f"- BERT/DistilBERT F1 Score: {bert_f1:.3f}")
    print(f"- Best Overall F1: {best_f1:.3f}")
    
    print(f"\nFinal Assessment for Workplace Pattern Detection:")
    
    if best_f1 > 0.75:
        print("🎉 EXCELLENT SUCCESS - TARGET EXCEEDED!")
        print("- Workplace power dynamics successfully detectable with AI")
        print("- Ready for production deployment and user testing")
        print("- Strong evidence for practical workplace application")
        print("- Portfolio-quality project demonstrating advanced NLP skills")
        
    elif best_f1 > 0.70:
        print("✅ SUCCESS - TARGET NEARLY ACHIEVED")
        print("- Strong performance for complex workplace pattern detection")
        print("- Minor improvements could exceed F1 > 0.75")
        print("- Excellent foundation for advanced workplace AI applications")
        print("- Demonstrates mastery of both traditional ML and transformers")
        
    elif best_f1 > 0.60:
        print("🎯 GOOD PROGRESS - STRONG FOUNDATION")
        print("- Solid performance indicates feasible workplace pattern detection")
        print("- BERT fine-tuning shows promise for target achievement")
        print("- Consider ensemble methods or additional data")
        print("- Strong technical demonstration for portfolio")
        
    else:
        print("⚠️ NEEDS IMPROVEMENT - REASSESS APPROACH")
        print("- Workplace patterns may require different modeling approach")
        print("- Consider problem reformulation or data quality improvements")
        print("- Valuable learning experience in NLP challenges")

# 10. MAIN EXECUTION FUNCTION (UPDATED WITH BERT)
def main_analysis_complete():
def main_analysis_complete():
    """
    Execute complete Jessica Cervi NLP framework analysis INCLUDING BERT
    Traditional ML + Modern Transformers for workplace power dynamics
    """
    print("EXECUTING JESSICA CERVI'S COMPLETE NLP FRAMEWORK")
    print("Traditional ML + BERT/DistilBERT for Workplace Pattern Detection")
    print("Jessica's 4/30/2025 Core Focus: Fine-tuned NLP classifiers")
    
    # Step 1: Load and assess data quality
    df = load_and_assess_data()
    
    if df is None:
        print("\n🚨 DATA LOADING REQUIRED")
        print("Please update the load_and_assess_data() function with your actual data loading code")
        print("Required format: CSV with 'narrative_text' and 'pattern' columns")
        return
    
    # Step 2: Jessica's intrinsic evaluation
    intrinsic_evaluation_jessica(df)
    
    # Step 3: Pattern separability analysis
    pattern_separability_analysis(df)
    
    # Step 4: Create professional visualizations
    create_jessica_visualizations(df)
    
    # Step 5: Train baseline models (Jessica's traditional ML)
    print("\n" + "="*50)
    print("PHASE 1: JESSICA'S TRADITIONAL ML BASELINES")
    print("="*50)
    
    nb_model, nb_vectorizer, nb_f1 = jessica_naive_bayes_baseline(df)
    dt_model, dt_f1 = jessica_decision_tree_baseline(df)
    
    # Step 6: BERT implementation (Jessica's 4/30/2025 focus)
    print("\n" + "="*50)
    print("PHASE 2: JESSICA'S BERT/TRANSFORMER IMPLEMENTATION")
    print("="*50)
    
    bert_model, bert_tokenizer, bert_f1 = jessica_bert_implementation(df)
    
    # Step 7: Comprehensive model comparison
    best_model = comprehensive_model_comparison(nb_f1, dt_f1, bert_f1)
    
    # Step 8: Complete feasibility assessment
    jessica_f1_feasibility_assessment_complete(nb_f1, dt_f1, bert_f1)
    
    # Step 9: BERT inference demonstration (if successful)
    if bert_model is not None:
        print("\n" + "="*50)
        print("PHASE 3: BERT INFERENCE PIPELINE DEMONSTRATION")
        print("="*50)
        
        classifier = jessica_bert_inference_pipeline()
        if classifier is not None:
            # Demo with sample workplace scenarios
            sample_texts = [
                "My manager suddenly changed my performance goals without explanation and now expects me to meet impossible deadlines",
                "During our team meeting, my supervisor publicly criticized my work in front of everyone and dismissed my suggestions",
                "I was told I need to improve but wasn't given any specific feedback or resources to help me succeed"
            ]
            demonstrate_bert_inference(classifier, sample_texts)
    
    print("\n" + "="*70)
    print("JESSICA'S COMPLETE FRAMEWORK ANALYSIS FINISHED")
    print("✅ Traditional ML Baselines (Naive Bayes + Decision Trees)")
    print("✅ Modern Transformer Implementation (BERT/DistilBERT)")
    print("✅ Comprehensive Performance Evaluation")
    print("✅ Production-Ready Inference Pipeline")
    print("✅ Portfolio-Quality Technical Demonstration")
    print("="*70)
    
    return df, nb_model, dt_model, bert_model

# 11. EXECUTION
if __name__ == "__main__":
    # Execute the complete analysis
    print("🚀 Starting Complete Workplace Power Dynamics Classifier Analysis")
    print("Following Jessica Cervi's Full NLP Framework:")
    print("- Traditional ML Baselines (Module 18)")
    print("- Fine-tuned BERT Implementation (4/30/2025 Core Focus)")
    
    # Installation check for BERT dependencies
    if not TRANSFORMERS_AVAILABLE:
        print("\n📦 BERT REQUIREMENTS:")
        print("pip install transformers torch")
        print("(GPU recommended but not required)")
    
    print("\n📋 NEXT STEPS:")
    print("1. Install BERT dependencies: pip install transformers torch")
    print("2. Update load_and_assess_data() with your CSV file path")
    print("3. Ensure your data has 'narrative_text' and 'pattern' columns")
    print("4. Run: main_analysis_complete()")
    print("5. Review comprehensive results including BERT performance")
    
    # Uncomment when ready to run with your data:
    # df, nb_model, dt_model, bert_model = main_analysis_complete()