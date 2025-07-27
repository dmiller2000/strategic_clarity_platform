# Workplace Power Dynamics Classifier - Labeling Guidelines

## Overview
This document provides clear, consistent guidelines for labeling workplace narratives with power dynamic patterns. Use this as your reference for maintaining quality and consistency across all data labeling tasks.

## Core Principles

### 1. Single-Label Classification
- Assign **ONE primary pattern** per narrative
- If multiple patterns appear, choose the **most prominent/severe**
- Use "none" only if no clear pattern is present

### 2. Evidence-Based Labeling
- Pattern must have **clear textual evidence**
- Avoid assumptions about unstated context
- Focus on observable behaviors and explicit descriptions

### 3. Confidence Levels
- Only label examples where you feel **70%+ confident**
- When uncertain, mark for review or exclude

---

## Pattern Definitions & Guidelines

### 1. PIP_TACTICS
**Definition**: Performance Improvement Plans used as documentation for exit strategy rather than genuine development.

**Key Indicators to Look For:**
- ✅ Timing correlation with complaints/advocacy
- ✅ Vague, subjective performance criteria
- ✅ Contrast with previous positive performance
- ✅ Predetermined feeling vs. developmental intent
- ✅ Increased documentation requirements

**Example Language:**
- "suddenly put me on a PIP"
- "goals were vague"
- "previously positive reviews"
- "felt predetermined"
- "documenting every small interaction"

**NOT This Pattern:**
- ❌ Legitimate performance concerns with specific metrics
- ❌ Clear development plan with measurable goals
- ❌ First-time performance feedback

---

### 2. DOCUMENTATION_BUILDING
**Definition**: Systematic increase in written documentation of minor issues to build a case for future action.

**Key Indicators to Look For:**
- ✅ Sudden shift from verbal to written communication
- ✅ Documentation of previously ignored minor issues
- ✅ Increased formality in routine interactions
- ✅ Paper trail creation without operational necessity
- ✅ CC'ing HR/management on routine communications

**Example Language:**
- "now requires written confirmation"
- "started documenting everything"
- "emails about things we used to discuss verbally"
- "formal written warnings for minor issues"
- "creating a paper trail"

**NOT This Pattern:**
- ❌ Standard documentation practices
- ❌ Compliance-required written procedures
- ❌ New manager establishing consistent processes

---

### 3. ISOLATION_TACTICS
**Definition**: Systematic exclusion from meetings, information, or decision-making processes.

**Key Indicators to Look For:**
- ✅ Exclusion from relevant meetings without explanation
- ✅ Information withholding about projects/decisions
- ✅ Removal from communication channels
- ✅ Access restrictions to resources/systems
- ✅ Social/professional isolation from team

**Example Language:**
- "not invited to meetings"
- "excluded from planning sessions"
- "access was removed"
- "information being withheld"
- "left out of important discussions"

**NOT This Pattern:**
- ❌ Meetings outside person's scope of responsibility
- ❌ Confidential discussions requiring specific clearance
- ❌ Temporary access restrictions for technical reasons

---

### 4. STRATEGIC_AMBIGUITY
**Definition**: Deliberate vague communication creating deniability while limiting employee options.

**Key Indicators to Look For:**
- ✅ Contradictory guidance from same authority
- ✅ Refusal to provide written clarification
- ✅ Shifting success criteria without explanation
- ✅ "You should know" responses to clarification requests
- ✅ Verbal vs. written communication discrepancies

**Example Language:**
- "contradictory guidance"
- "won't put it in writing"
- "you should figure it out"
- "it should be obvious"
- "mixed messages"

**NOT This Pattern:**
- ❌ Genuinely unclear situations requiring collaborative problem-solving
- ❌ Complex decisions requiring time to formulate
- ❌ Evolving priorities due to external business factors

---

## Special Cases & Edge Cases

### Multiple Patterns Present
When narratives contain multiple patterns:
1. **Choose the most severe/prominent pattern**
2. **Consider temporal sequence** (which came first?)
3. **Focus on the pattern with strongest textual evidence**

**Priority Order** (most to least severe):
1. PIP_TACTICS
2. STRATEGIC_AMBIGUITY  
3. DOCUMENTATION_BUILDING
4. ISOLATION_TACTICS

### Borderline Cases
When unsure between two patterns:
- **PIP_TACTICS vs DOCUMENTATION_BUILDING**: Choose PIP if formal improvement plan is mentioned
- **STRATEGIC_AMBIGUITY vs poor communication**: Choose Strategic Ambiguity if intentionality is implied
- **ISOLATION_TACTICS vs organizational change**: Choose Isolation if targeting is evident

### Industry Variations
**Same pattern, different language:**
- **Tech**: "cultural fit," "agile adaptation," "stakeholder alignment"
- **Healthcare**: "patient safety," "protocol compliance," "professional standards"
- **Finance**: "risk management," "regulatory compliance," "audit readiness"
- **Education**: "student outcomes," "administrative alignment," "professional development"

---

## Quality Control Checklist

Before finalizing each label, verify:

- [ ] **Pattern is clearly evident** in the text
- [ ] **Key indicators are present** (minimum 2-3)
- [ ] **Alternative explanations** have been considered
- [ ] **Confidence level** is 70%+ 
- [ ] **Industry context** has been considered
- [ ] **Narrative length** is sufficient (100+ words preferred)

---

## Common Labeling Mistakes to Avoid

### ❌ Over-labeling
- Don't force patterns onto normal workplace issues
- Avoid labeling based on emotional tone alone
- Don't assume malicious intent without evidence

### ❌ Under-labeling  
- Don't dismiss patterns due to professional language
- Avoid missing subtle but consistent behavioral changes
- Don't require explicit admission of problematic behavior

### ❌ Inconsistent Standards
- Apply same criteria across all industries
- Maintain consistent confidence thresholds
- Use same evidence requirements for all patterns

---

## Inter-Rater Reliability

To ensure consistency:
1. **Start with 20 sample narratives** labeled by multiple people
2. **Calculate agreement rates** (target: 80%+ agreement)
3. **Discuss disagreements** and refine guidelines
4. **Retest with new sample** before proceeding

---

## Documentation Requirements

For each labeled example, track:
- **narrative_id**: Unique identifier
- **text**: Full workplace narrative
- **pattern**: One of 4 patterns or "none"  
- **source**: reddit/glassdoor/synthetic
- **confidence_notes**: Brief justification (optional)

---

## Quick Reference Card

| Pattern | Key Question | Primary Evidence |
|---------|--------------|------------------|
| **PIP_TACTICS** | "Is this genuine development or exit documentation?" | Timing + vague goals + previous positive performance |
| **DOCUMENTATION_BUILDING** | "Why the sudden shift to written records?" | Verbal→written shift + minor issues escalated |
| **ISOLATION_TACTICS** | "Is exclusion systematic and unexplained?" | Meeting exclusion + information withholding |
| **STRATEGIC_AMBIGUITY** | "Is confusion deliberate or circumstantial?" | Contradictory guidance + refusal to clarify |

---

## When in Doubt

**If uncertain about a label:**
1. **Mark for review** rather than guessing
2. **Exclude from dataset** if confidence <70%
3. **Consult with second labeler** for difficult cases
4. **Document reasoning** for future reference

Remember: **Quality over quantity**. Better to have 200 high-confidence labels than 300 uncertain ones.