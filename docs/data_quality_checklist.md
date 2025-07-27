Data Quality Checklist - Clarity Engine Dataset

## Overview
This checklist ensures high-quality data collection, processing, and validation for the Clarity Engine: AI-Powered Workplace Navigation Platform. Use this at each stage of your data pipeline to maintain professional standards.

---

## Pre-Collection Setup ✅

### Data Structure Validation
- [ ] **CSV template created** with correct headers: `narrative_id,text,pattern,source`
- [ ] **Folder structure established**: `raw/`, `processed/`, `data_template_ideas/`
- [ ] **Labeling guidelines documented** and accessible
- [ ] **Target quotas defined**: 75 examples per pattern (300 total)
- [ ] **Collection timeline established** with daily targets

### Ethical & Legal Compliance
- [ ] **Data collection methods reviewed** for ToS compliance
- [ ] **Anonymization strategy planned** for personal information
- [ ] **Consent considerations addressed** for user-generated content
- [ ] **Data retention policies established**
- [ ] **Privacy protection measures implemented**

---

## During Collection 📥

### Source Quality Standards
- [ ] **Narrative length adequate** (100-500 words preferred)
- [ ] **Clear workplace context** present in text
- [ ] **Sufficient detail** for pattern identification
- [ ] **English language** content only
- [ ] **Recent timeframe** (last 5 years preferred)

### Content Validation Per Example
- [ ] **Pattern evidence present** (minimum 2-3 indicators)
- [ ] **No identifying information** (names, companies, locations)
- [ ] **Coherent narrative structure** with clear sequence
- [ ] **Workplace setting confirmed** (not personal relationships)
- [ ] **Professional language appropriate** for analysis

### Real-Time Quality Checks
- [ ] **Duplicate detection** before adding to dataset
- [ ] **Spam/fake content filtering** applied
- [ ] **Label confidence assessment** (>70% required)
- [ ] **Source diversity maintained** across platforms
- [ ] **Pattern distribution monitored** for balance

---

## Post-Collection Processing 🔄

### Data Completeness
- [ ] **All required fields populated** (no empty cells)
- [ ] **Narrative IDs sequential** and unique (1, 2, 3...)
- [ ] **Pattern labels standardized** (exact spelling: `pip_tactics`, `documentation_building`, `isolation_tactics`, `strategic_ambiguity`)
- [ ] **Source labels consistent** (`reddit`, `glassdoor`, `synthetic`, `other`)
- [ ] **Target sample sizes achieved** per pattern

### Text Quality Assessment
- [ ] **Character encoding consistent** (UTF-8)
- [ ] **Special characters handled** properly
- [ ] **Line breaks/formatting standardized**
- [ ] **Excessive whitespace removed**
- [ ] **Text length distribution analyzed**

### Anonymization Verification
- [ ] **Personal names removed** or pseudonymized
- [ ] **Company names generalized** (e.g., "Fortune 500 tech company")
- [ ] **Geographic specifics removed** (cities, states)
- [ ] **Department/role titles generalized** when identifying
- [ ] **Timeline details obfuscated** if necessary

---

## Statistical Quality Analysis 📊

### Dataset Distribution
- [ ] **Pattern balance assessed**: Target 25% per pattern
- [ ] **Source distribution analyzed**: No single source >60%
- [ ] **Text length distribution reviewed**: Mean 200-400 words
- [ ] **Industry representation evaluated**: Diverse sectors included
- [ ] **Temporal distribution checked**: Recent examples prioritized

### Pattern Quality Metrics
- [ ] **Inter-rater reliability tested**: >80% agreement target
- [ ] **Label confidence distribution**: >70% confidence minimum
- [ ] **Pattern distinctiveness verified**: Clear separability
- [ ] **Co-occurrence patterns documented**: Expected relationships
- [ ] **Edge case representation**: Borderline examples included

### Text Analysis Metrics
- [ ] **Vocabulary diversity assessed**: Rich language variety
- [ ] **Reading level appropriate**: Professional but accessible
- [ ] **Sentiment distribution analyzed**: Range of emotional tones
- [ ] **Keyword frequency checked**: Pattern-specific terms present
- [ ] **Language complexity measured**: Suitable for NLP processing

---

## Pre-Processing Pipeline ⚙️

### Data Cleaning Steps
- [ ] **Duplicate removal completed**: Exact and near-duplicate detection
- [ ] **Text normalization applied**: Consistent formatting
- [ ] **Invalid entries filtered**: Too short, incoherent, off-topic
- [ ] **Quality scoring implemented**: Systematic quality assessment
- [ ] **Manual review completed**: Spot-checking random samples

### Feature Engineering Preparation
- [ ] **Text preprocessing planned**: Tokenization, cleaning strategy
- [ ] **Stop words list customized**: Domain-specific additions
- [ ] **Stemming/lemmatization decided**: Approach selected
- [ ] **N-gram analysis prepared**: Unigram/bigram strategy
- [ ] **Domain vocabulary identified**: Workplace-specific terms

### Train/Test Split Preparation
- [ ] **Stratified sampling planned**: Equal pattern representation
- [ ] **Random seed set**: Reproducible splits
- [ ] **Validation strategy chosen**: Cross-validation approach
- [ ] **Test set size determined**: 20-25% of total data
- [ ] **Data leakage prevention**: No contamination between sets

---

## Final Validation Checks ✅

### Technical Validation
- [ ] **File format verified**: Proper CSV structure
- [ ] **Encoding confirmed**: UTF-8 without BOM
- [ ] **Headers correct**: Exact column names match template
- [ ] **Data types appropriate**: Text fields properly formatted
- [ ] **File size reasonable**: Not corrupted or truncated

### Content Validation
- [ ] **Random sample manual review**: 10% of dataset checked
- [ ] **Pattern representation verified**: All 4 patterns present
- [ ] **Quality standards maintained**: Consistent throughout
- [ ] **Labeling accuracy confirmed**: Guidelines followed
- [ ] **Edge cases included**: Variety of manifestations

### Documentation Completeness
- [ ] **Data dictionary created**: Column definitions clear
- [ ] **Collection methodology documented**: Sources and methods
- [ ] **Quality metrics recorded**: Statistical summaries
- [ ] **Known limitations identified**: Bias and constraints
- [ ] **Processing steps logged**: Reproducible pipeline

---

## EDA Preparation Checklist 📈

### Priority #1: Data Quality Documentation
- [ ] **Missing value analysis**: Complete assessment
- [ ] **Duplicate detection results**: Counts and examples
- [ ] **Text cleaning pipeline documented**: Before/after examples
- [ ] **Quality metrics visualized**: Distribution plots
- [ ] **Processing impact measured**: Changes quantified

### Priority #2: Pattern Feasibility Analysis
- [ ] **Pattern distribution visualized**: Bar charts and statistics
- [ ] **Vocabulary analysis completed**: Pattern-specific word clouds
- [ ] **TF-IDF analysis prepared**: Feature importance ready
- [ ] **Pattern separability assessed**: Clustering visualization
- [ ] **F1 >0.75 feasibility demonstrated**: Evidence provided

### Professional Presentation Ready
- [ ] **Clean, professional visualizations**: Publication-quality plots
- [ ] **Statistical summaries prepared**: Key metrics highlighted
- [ ] **Methodology clearly explained**: Transparent process
- [ ] **Limitations acknowledged**: Honest assessment
- [ ] **Next steps outlined**: Modeling approach planned

---

## Emergency Quality Indicators ⚠️

### Red Flags (Stop and Fix)
- ❌ **>30% missing/invalid data** in any column
- ❌ **Pattern imbalance >60/40** between categories
- ❌ **>20% duplicate content** across dataset
- ❌ **Significant identifying information** still present
- ❌ **Inter-rater agreement <70%** on sample

### Yellow Flags (Monitor Closely)
- ⚠️ **Single source >50%** of total data
- ⚠️ **Average text length <150 words**
- ⚠️ **Pattern confidence <75%** on sample
- ⚠️ **Limited industry diversity** in examples
- ⚠️ **Processing errors >5%** of dataset

### Green Flags (Quality Targets)
- ✅ **Pattern balance 20-30%** each category
- ✅ **Source diversity** across platforms
- ✅ **High label confidence** >80% average
- ✅ **Rich text content** 200-400 word average
- ✅ **Strong pattern separability** in analysis

---

## Daily Quality Monitoring (9-Day Sprint) 🗓️

### Daily Checkpoints
- [ ] **Day 1-2**: Template and guidelines quality verified
- [ ] **Day 3-4**: Collection quality monitored real-time
- [ ] **Day 5**: Mid-point quality assessment completed
- [ ] **Day 6-7**: Processing quality validated
- [ ] **Day 8-9**: Final quality confirmation before analysis

### Daily Metrics to Track
- **Examples collected**: Progress toward 300 total
- **Quality rejection rate**: <20% target
- **Pattern distribution**: Balance maintained
- **Processing errors**: <5% of daily collection
- **Time per example**: Efficiency monitoring

---

## Final Sign-Off Checklist ✍️

Before proceeding to EDA and modeling:

- [ ] **All quality checks passed** above thresholds
- [ ] **Documentation complete** and professional
- [ ] **Data stored securely** with backups
- [ ] **Processing pipeline documented** for reproducibility
- [ ] **Team review completed** (if applicable)
- [ ] **Jessica's priorities addressed**: Data quality + pattern feasibility
- [ ] **Timeline met**: Ready for EDA phase
- [ ] **Confidence high**: Dataset suitable for ML analysis

**Quality Assurance Lead**: _________________ **Date**: _________

**Final Dataset Approved**: ✅ **Ready for EDA and Modeling**
