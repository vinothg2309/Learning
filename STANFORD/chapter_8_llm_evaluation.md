chapter_8_llm_evaluation.md
# Resources
<!-- TOC -->
## Table of Contents

- [Resources](#resources)
  - [Table of Contents](#table-of-contents)
- [Evaluation](#evaluation)
  - [Limitation - Human Rating](#limitation---human-rating)
- [Rule Based Metrics](#rule-based-metrics)
  - [METEOR - Metric for Evaluation of Translation with Explicit ORdering](#meteor---metric-for-evaluation-of-translation-with-explicit-ordering)
  - [BLEU (BiLingual Evaluation Understudy)](#bleu-bilingual-evaluation-understudy)
  - [ROUGE (Recall-Oriented Understudy for Gisting Evaluation)](#rouge-recall-oriented-understudy-for-gisting-evaluation)
  - [Limitation](#limitation)
- [LLM as a Judge (LaaJ)](#llm-as-a-judge-laaj)
  - [Benefits:](#benefits)
  - [Variation](#variation)
    - [Pointwise](#pointwise)
    - [Pairwise](#pairwise)
    - [Position bias](#position-bias)
    - [Verbosity bias](#verbosity-bias)
    - [Self-enhancement Bias](#self-enhancement-bias)
    - [Best Practices](#best-practices)
    - [Dimension to evaluate](#dimension-to-evaluate)
      - [Factuality](#factuality)
- [Agent Evaluation](#agent-evaluation)
  - [Tool Prediction Error](#tool-prediction-error)
  - [Hallucination Tool](#hallucination-tool)
  - [Using wrong tool](#using-wrong-tool)
  - [Infers wrong arguments](#infers-wrong-arguments)
  - [Wrong Response](#wrong-response)
  - [No Response](#no-response)
- [Benchmarks](#benchmarks)
  - [Common Benchmarks](#common-benchmarks)
  - [MMLU (Massive Multitask Language Understanding) - Knowledge](#mmlu-massive-multitask-language-understanding---knowledge)
  - [Reasoning](#reasoning)
    - [AIME (Math)](#aime-math)
    - [PIQA](#piqa)
  - [Coding](#coding)
    - [SWE](#swe)
  - [Safety](#safety)
    - [HarmBench](#harmbench)
  - [Agent Benchmarking](#agent-benchmarking)
    - [Tau-Bench(𝜏-bench)](#tau-bench𝜏-bench)
  - [Pareto frontier](#pareto-frontier)
  - [Data contamination](#data-contamination)
<!-- /TOC -->

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

# Evaluation

![alt text](images/evaluation_intro.png)

*This image introduces the concept of LLM evaluation - the process of measuring how well a language model performs. It likely covers why evaluation is crucial for understanding model quality, reliability, and real-world performance.*

![alt text](images/evaluation_approaches.png)

*This image explains different evaluation approaches for LLMs. There are typically two main categories: **automatic evaluation** (using metrics and algorithms) and **human evaluation** (where people judge the quality of responses). Each method has its own strengths and use cases.*

## Limitation - Human Rating

`Each user have their own preference in response`

![alt text](images/human_rating_limitation.png)

*This image highlights a key challenge with human evaluation: **subjectivity**. Different people may rate the same LLM response differently based on their personal preferences, background, or expectations. What one person considers a "good" answer, another might find inadequate. This makes it hard to get consistent, reliable evaluation scores.*

![alt text](images/inter_rater_agreement.png)

*This image introduces **inter-rater agreement metrics** - statistical measures used to check if multiple human evaluators agree with each other when rating LLM outputs. Common metrics include **Cohen's Kappa** (for two raters) and **Fleiss' Kappa** (for three or more raters). These metrics help determine if human ratings are consistent and reliable. High agreement means evaluators are rating similarly; low agreement suggests the evaluation criteria might be unclear or too subjective.*

**Cohen's Kappa**: A statistical measure that checks agreement between **two evaluators**. It ranges from -1 to 1, where 1 means perfect agreement, 0 means agreement by random chance, and negative values mean worse than random agreement.

*Step-by-Step Example*:
1. **Setup**: Alice and Bob evaluate 10 chatbot responses, rating each as "Good" or "Bad"
2. **Results**: 
   - Response 1: Alice=Good, Bob=Good ✓ (Agree)
   - Response 2: Alice=Good, Bob=Bad ✗ (Disagree)
   - Response 3: Alice=Bad, Bob=Bad ✓ (Agree)
   - ...continuing for all 10 responses
3. **Count Agreement**: They agreed on 8 out of 10 responses (80% observed agreement)
4. **Calculate Expected Agreement**: By random chance, we'd expect them to agree ~50% of the time
5. **Cohen's Kappa Result**: κ = (80% - 50%) / (100% - 50%) = 0.60
6. **Interpretation**: 0.60 indicates "moderate agreement" - they're rating consistently, but not perfectly

**Fleiss' Kappa**: Similar to Cohen's Kappa, but works for **three or more evaluators**. It measures how much agreement exists among multiple raters beyond what we'd expect by chance.

*Step-by-Step Example*:
1. **Setup**: Three judges (Amy, Ben, Carl) rate 5 LLM summaries as "Accurate", "Neutral", or "Inaccurate"
2. **Results for Summary 1**:
   - Amy=Accurate, Ben=Accurate, Carl=Accurate (Perfect agreement!)
3. **Results for Summary 2**:
   - Amy=Neutral, Ben=Accurate, Carl=Neutral (Partial agreement - 2 out of 3 agree)
4. **Results for Summary 3**:
   - Amy=Inaccurate, Ben=Inaccurate, Carl=Inaccurate (Perfect agreement!)
5. **Results for Summaries 4 & 5**: Mixed agreements
6. **Calculate**: Fleiss' Kappa looks at all ratings together and compares actual agreement vs. random chance
7. **Result**: If κ = 0.72, this means "substantial agreement" - the judges have clear, shared criteria
8. **Interpretation**: High Kappa (>0.7) = reliable evaluation; Low Kappa (<0.4) = need clearer guidelines


![alt text](images/kappa_interpretation.png)

*This image likely shows interpretation guidelines for Kappa values, helping you understand what different scores mean. Typically: < 0.20 = slight agreement, 0.21-0.40 = fair, 0.41-0.60 = moderate, 0.61-0.80 = substantial, 0.81-1.00 = almost perfect agreement.*

# Rule Based Metrics

## METEOR - Metric for Evaluation of Translation with Explicit ORdering

**What is METEOR?**
METEOR is an automatic evaluation metric originally designed for machine translation, but now widely used for evaluating LLM-generated text. Unlike simple word-matching metrics, METEOR is smarter because it considers:
- **Exact matches** (same words)
- **Synonyms** (words with similar meanings)
- **Stemming** (word variations like "run", "running", "runs")
- **Word order** (sequence of words matters)

![alt text](images/meteor_components.png)

*This image likely explains the components of METEOR and how it calculates scores by considering precision (how much of the generated text is correct) and recall (how much of the reference text is captured).*

![alt text](images/meteor_formula.png)

*This image probably shows the METEOR formula or scoring process, demonstrating how it combines different matching types and applies a penalty for incorrect word ordering.*

**Step-by-Step Example**:

**Scenario**: Evaluating an LLM's summary against a reference summary

1. **Reference (Human-written)**: "The cat quickly ran across the street"
   - Total words: 7
2. **Generated (LLM output)**: "The big feline rapidly runs over the road yesterday"
   - Total words: 9

**Step 1: Find Exact Matches**
- "The" matches "The" ✓
- No other exact word matches

**Step 2: Find Synonym Matches**
- "cat" ≈ "feline" ✓ (synonym)
- "quickly" ≈ "rapidly" ✓ (synonym)
- "across" ≈ "over" ✓ (synonym)
- "street" ≈ "road" ✓ (synonym)

**Step 3: Find Stem Matches**
- "ran" ≈ "runs" ✓ (same stem: "run")

**Step 4: Calculate Precision & Recall**
- **Total Matches Found**: 6 words matched (The, cat/feline, quickly/rapidly, ran/runs, across/over, street/road)
- **Precision**: How many words in the generated text matched the reference?
  - 6 matched out of 9 generated words = 6/9 = **66.7%**
  - (Lower because LLM added extra words: "big" and "yesterday")
- **Recall**: How many words from the reference were found in generated text?
  - 6 matched out of 7 reference words = 6/7 = **85.7%**
  - (Higher because most reference words were captured)

**Step 5: Calculate F-mean**
- Combines precision and recall into one score
- F-mean = (10 × Precision × Recall) / (9 × Precision + Recall)

**Step 6: Apply Word Order Penalty**
- Check if matched words appear in the same order
- "The feline rapidly runs over the road" vs "The cat quickly ran across the street"
- Words are mostly in order, so penalty is small (e.g., 0.05)

**Step 7: Final METEOR Score**
- Final Score = F-mean × (1 - Penalty)
- Example: 0.857 × (1 - 0.05) = **0.814**

**Interpretation**:
- Score ranges from 0 to 1
- **0.814 is high** = LLM output is very similar to reference in meaning and structure
- Higher score = better match with reference text
- METEOR is better than simple word matching because it understands synonyms and word variations!

---

## BLEU (BiLingual Evaluation Understudy)

**What is BLEU?**
BLEU is one of the most popular automatic evaluation metrics for machine translation and text generation. Unlike METEOR, BLEU is simpler and focuses on **n-gram precision** - checking how many word sequences (1-word, 2-word, 3-word, 4-word phrases) from the generated text appear in the reference text.

**Key Concepts:**
- **N-grams**: Sequences of N consecutive words
  - 1-gram (unigram): individual words like "the", "cat"
  - 2-gram (bigram): pairs like "the cat", "ran quickly"
  - 3-gram (trigram): triplets like "the cat ran"
  - 4-gram: four-word sequences
- **Precision-focused**: BLEU mainly measures if generated words match the reference (doesn't heavily penalize missing content)
- **Brevity Penalty**: Punishes outputs that are too short

![alt text](images/bleu_formula.png)

*This image likely shows the BLEU formula and how it combines different n-gram precisions (1-gram through 4-gram) with a brevity penalty to produce a final score between 0 and 1.*

**Step-by-Step Example**:

**Scenario**: Evaluating a translation or LLM output

**Reference**: "The cat sat on the mat"
**Generated**: "The cat sat on a mat"

**Step 1: Count 1-grams (Individual Words)**
- Generated words: "The", "cat", "sat", "on", "a", "mat" (6 words)
- Matches in reference:
  - "The" ✓, "cat" ✓, "sat" ✓, "on" ✓, "a" ✗ (reference has "the" not "a"), "mat" ✓
- **1-gram precision**: 5 matches / 6 words = **5/6 = 83.3%**

**Step 2: Count 2-grams (Word Pairs)**
- Generated 2-grams: "The cat", "cat sat", "sat on", "on a", "a mat" (5 bigrams)
- Matches in reference:
  - "The cat" ✓, "cat sat" ✓, "sat on" ✓, "on a" ✗ (reference has "on the"), "a mat" ✗ (reference has "the mat")
- **2-gram precision**: 3 matches / 5 bigrams = **3/5 = 60%**

**Step 3: Count 3-grams (Word Triplets)**
- Generated 3-grams: "The cat sat", "cat sat on", "sat on a", "on a mat" (4 trigrams)
- Matches in reference:
  - "The cat sat" ✓, "cat sat on" ✓, "sat on a" ✗, "on a mat" ✗
- **3-gram precision**: 2 matches / 4 trigrams = **2/4 = 50%**

**Step 4: Count 4-grams (Four-word Sequences)**
- Generated 4-grams: "The cat sat on", "cat sat on a", "sat on a mat" (3 four-grams)
- Matches in reference:
  - "The cat sat on" ✓, "cat sat on a" ✗, "sat on a mat" ✗
- **4-gram precision**: 1 match / 3 four-grams = **1/3 = 33.3%**

**Step 5: Apply Brevity Penalty**
- Reference length: 6 words
- Generated length: 6 words
- Since lengths are equal, **Brevity Penalty (BP) = 1** (no penalty)
- If generated was shorter (e.g., 4 words), BP would be < 1 (penalty applied)

**Step 6: Calculate Final BLEU Score**
- BLEU combines all n-gram precisions using geometric mean:
- BLEU = BP × (1-gram × 2-gram × 3-gram × 4-gram)^(1/4)
- BLEU = 1 × (0.833 × 0.60 × 0.50 × 0.333)^0.25
- BLEU = (0.0833)^0.25 ≈ **0.538** or **53.8%**

**Interpretation**:
- **Score range**: 0 to 1 (often shown as 0-100)
- **0.538 (53.8%)** = Moderate match - one word difference significantly impacted score
- **Higher scores** (>0.7) = very good match with reference
- **Lower scores** (<0.3) = poor match, many differences
- **BLEU is strict**: Even small changes (like "the" → "a") significantly reduce the score!

**Key Difference from METEOR**:
- BLEU doesn't understand synonyms (strict word matching only)
- BLEU focuses on precision (good matches) rather than recall (coverage)
- BLEU is simpler and faster, but less forgiving than METEOR

---

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**What is ROUGE?**
ROUGE is an evaluation metric primarily used for **summarization tasks**. While BLEU focuses on precision (how accurate is the generated text?), ROUGE focuses on **recall** (how much of the reference content is captured?). This makes ROUGE perfect for evaluating summaries where you want to ensure important information isn't missed.

**Key Concepts:**
- **Recall-focused**: Measures how much of the reference text appears in the generated summary
- **Multiple variants**:
  - **ROUGE-N**: Counts overlapping n-grams (like BLEU, but recall-based)
  - **ROUGE-L**: Looks for the longest common subsequence (word order matters)
  - **ROUGE-S**: Considers skip-bigrams (allows gaps between words)

![alt text](images/rouge_variants.png)

*This image likely shows the different ROUGE variants (ROUGE-1, ROUGE-2, ROUGE-L) and their formulas, emphasizing how they measure recall to ensure generated summaries capture key information from reference texts.*

**Simple Example:**

**Reference Summary**: "The quick brown fox jumps over the lazy dog"
**Generated Summary**: "The brown fox jumps over the dog"

**ROUGE-1 (Unigram Recall)**:
- Reference has 9 words: "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
- Generated captured 7 of these words (missing "quick" and "lazy")
- **ROUGE-1 Recall** = 7/9 = **77.8%**

**ROUGE-2 (Bigram Recall)**:
- Reference bigrams: "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog" (8 bigrams)
- Generated bigrams matching reference: "brown fox", "fox jumps", "jumps over", "over the" (4 bigrams match)
- **ROUGE-2 Recall** = 4/8 = **50%**

**Interpretation:**
- **High ROUGE-1** (77.8%) = Most important words are captured
- **Lower ROUGE-2** (50%) = Some word sequences are missing
- Higher ROUGE scores = better coverage of reference content
- ROUGE is ideal for summaries where completeness matters!

**Key Difference:**
- **BLEU**: "Is the generated text accurate?" (Precision)
- **ROUGE**: "Did we capture the important content?" (Recall)
- **Use BLEU for**: Translation, text generation
- **Use ROUGE for**: Summarization, content coverage

---

## Limitation

![alt text](images/rule_based_limitations.png)

*This image likely illustrates the key limitations of rule-based metrics (METEOR, BLEU, ROUGE) when evaluating LLM outputs, emphasizing why these metrics alone are insufficient for comprehensive evaluation.*

**Limitations of METEOR, BLEU, and ROUGE:**

1. **No Understanding of Meaning**
   - These metrics only match words/phrases, not meaning
   - Example: "The movie was not bad" vs "The movie was good" - different meanings, but high word overlap!
   - Can't detect if the answer is actually correct or just uses similar words

2. **Cannot Judge Quality or Helpfulness**
   - High score ≠ helpful response
   - A response can match reference text perfectly but miss the user's actual intent
   - Don't measure creativity, clarity, or usefulness

3. **Require Reference Texts**
   - Need "gold standard" human-written references to compare against
   - What if there are multiple valid answers? (common in open-ended questions)
   - Real-world LLM applications often don't have reference texts

4. **Strict Word Matching (BLEU/ROUGE)**
   - Penalize valid paraphrases and synonyms heavily
   - "car" vs "automobile" - same meaning, but treated as different
   - METEOR helps with synonyms, but still limited

5. **Don't Catch Hallucinations or Factual Errors**
   - Can't verify if information is true or fabricated
   - Example: LLM says "Paris is in Germany" with perfect grammar - metrics won't catch this error!

6. **Ignore Context and Coherence**
   - Don't check if the response makes logical sense
   - Can't evaluate tone, style, or appropriateness for the situation
   - Miss whether the answer actually addresses the question

7. **Poor for Creative or Open-ended Tasks**
   - Terrible for evaluating stories, poems, or creative writing
   - Multiple "correct" answers exist - metrics favor only the reference
   - Limit innovation by penalizing different but valid approaches

**Bottom Line:**
Rule-based metrics are **quick and cheap** but **limited**. They work best for tasks with clear reference answers (like translation), but struggle with the complexity and creativity of modern LLM applications. This is why we need more advanced evaluation methods like "LLM as Judge"!

---
---

# LLM as a Judge (LaaJ)

**What is LLM as a Judge?**
Instead of using rule-based metrics (METEOR, BLEU, ROUGE) or expensive human evaluators, we use **another LLM** to evaluate the quality of LLM outputs. Think of it like having an AI teacher grade AI student's homework! The judge LLM reads the response and provides scores or feedback based on specific criteria like accuracy, helpfulness, coherence, and relevance.

**Why Use LLM as a Judge?**
- **Understands meaning**: Unlike word-matching metrics, it comprehends context and semantics
- **Scalable**: Can evaluate thousands of responses quickly and cheaply
- **Flexible**: Can judge any criteria (creativity, tone, factual accuracy, etc.)
- **No reference needed**: Can evaluate without requiring gold-standard answers
- **Cost-effective**: Cheaper than hiring human evaluators

![alt text](images/llm_as_judge_intro.png)

*This image introduces the concept of LLM as a Judge - using a powerful language model (like GPT-4) to automatically evaluate other LLM outputs. It likely shows the basic workflow: feeding both the prompt and response to a judge LLM, which then provides quality ratings or scores.*

![alt text](images/llm_judge_evaluation_modes.png)

*This image probably demonstrates different evaluation modes for LLM judges: **pointwise evaluation** (judge rates one response independently), **pairwise evaluation** (judge compares two responses and picks the better one), and **listwise evaluation** (judge ranks multiple responses). Pairwise is most common as it's easier for LLMs to compare than assign absolute scores.*

![alt text](images/llm_judge_prompts.png)

*This image likely shows example prompts used for LLM judges, including the evaluation criteria, scoring rubric, and instructions. A good judge prompt clearly defines what to evaluate (e.g., "Rate accuracy, helpfulness, and clarity on a scale of 1-5") and provides examples of good vs bad responses.*

![alt text](images/llm_judge_output.png)

*This image probably illustrates the output format from an LLM judge - typically includes numerical scores for different criteria, reasoning/explanation for the scores, and sometimes specific feedback. For example: "Accuracy: 4/5 - Response is mostly correct but missed one detail. Helpfulness: 5/5 - Directly answers the question."*

![alt text](images/llm_judge_limitations.png)

*This image likely discusses limitations and challenges of LLM as a Judge: **position bias** (prefers first/last responses), **self-preference bias** (rates its own outputs higher), **verbosity bias** (favors longer responses), and **inconsistency** (same response may get different scores). These biases need to be mitigated through careful prompt design and multiple evaluation rounds.*

**Key Advantages over Rule-Based Metrics:**
- ✅ Understands semantic meaning, not just word overlap
- ✅ Can catch hallucinations and factual errors
- ✅ Evaluates coherence, tone, and appropriateness
- ✅ Works great for creative and open-ended tasks
- ✅ No reference text required

**Common Use Cases:**
- Comparing multiple LLM models (which is better?)
- A/B testing different prompts or configurations
- Monitoring production LLM quality over time
- Evaluating chatbot conversations for helpfulness
- Judging creative outputs like stories or marketing copy

**Popular Judge LLMs:**
- GPT-4, GPT-4o (OpenAI)
- Claude (Anthropic)
- PaLM 2 (Google)
- Generally: Stronger models = better judges!

## Benefits:

![alt text](images/llm_judge_benefits.png)

*This image likely summarizes the key benefits of using LLM as a Judge: scales better than human evaluation, understands context and meaning (unlike rule-based metrics), flexible for various evaluation criteria, cost-effective, and can work without reference texts.*

## Variation

**LLM as a Judge** has different evaluation approaches depending on how you present responses to the judge. Each method has its own strengths and use cases.

![alt text](images/llm_judge_variation.png)

### Pointwise

**What is Pointwise Evaluation?**
The judge LLM evaluates **one response at a time** independently, assigning an absolute score without comparing it to other responses.

**How it works:**
1. Give the judge: Question + Single Response
2. Judge rates it on a scale (e.g., 1-5 or 1-10)
3. Provides scores for specific criteria (accuracy, helpfulness, clarity)

**Example:**
- **Question**: "What is photosynthesis?"
- **Response A**: "Photosynthesis is how plants make food using sunlight."
- **Judge's Rating**: "Accuracy: 4/5, Clarity: 5/5, Completeness: 3/5"

**Pros:**
- ✅ Simple and straightforward
- ✅ Can evaluate any number of responses independently
- ✅ Gives absolute quality scores

**Cons:**
- ❌ Inconsistent scoring (same quality response may get different scores)
- ❌ Difficult for LLMs to assign absolute numbers
- ❌ Scores can vary based on judge's "mood" or context

**Best for:** Monitoring quality over time, getting quick quality estimates

### Pairwise

**What is Pairwise Evaluation?**
The judge LLM compares **two responses side-by-side** and determines which one is better. Instead of assigning absolute scores, it makes relative judgments.

**How it works:**
1. Give the judge: Question + Response A + Response B
2. Judge picks which response is better (A or B)
3. Provides reasoning for the choice

**Example:**
- **Question**: "What is photosynthesis?"
- **Response A**: "Photosynthesis is how plants make food using sunlight."
- **Response B**: "Photosynthesis is the process where plants convert light energy, water, and CO2 into glucose and oxygen."
- **Judge's Decision**: "Response B is better - more detailed and scientifically accurate."

**Pros:**
- ✅ More reliable and consistent than pointwise
- ✅ Easier for LLMs to compare than assign absolute scores
- ✅ Reduces scoring inconsistency
- ✅ Most commonly used in practice

**Cons:**
- ❌ Requires many comparisons for multiple responses (N responses = N×(N-1)/2 comparisons)
- ❌ Position bias (may favor first or second position)
- ❌ Doesn't give absolute quality scores

**Best for:** A/B testing, model comparison, ranking different LLM outputs

**Quick Comparison:**
| Aspect | Pointwise | Pairwise |
|--------|-----------|----------|
| Input | 1 response | 2 responses |
| Output | Absolute score | Which is better |
| Consistency | Lower | Higher |
| Speed | Faster | Slower (more calls) |
| Use Case | Quality monitoring | Model comparison |

**Note:** There's also **Listwise** evaluation (judge ranks 3+ responses at once), but it's less common and harder for LLMs to do accurately.

### Position bias

**What is Position Bias?**
Position bias is a common problem in LLM judges where the judge **unfairly favors responses based on their position** (first, second, or last) rather than their actual quality. This is especially problematic in pairwise comparisons.

**The Problem:**
LLM judges often show systematic preference for:
- **First position** (primacy bias) - favors Response A
- **Last position** (recency bias) - favors Response B
- This happens even when both responses have equal quality!

**Example:**
- **Test 1**: Judge sees "Response A | Response B" → Picks A (60% of the time)
- **Test 2**: Same responses, but flipped "Response B | Response A" → Picks A again (60% of the time)
- **Problem**: If truly unbiased, Response A should win ~50% regardless of position

![alt text](images/position_bias.png)

*This image likely demonstrates position bias in action, showing how judge LLMs exhibit different win rates for the same response depending on whether it appears first or second in the comparison. It may include statistics showing the bias magnitude (e.g., "Response in first position wins 65% of the time vs expected 50%").*

**Why Does This Happen?**
- LLMs are trained on text where earlier/later information may have different importance
- Attention mechanisms may weight certain positions differently
- Judge may be more influenced by first impression or final impression

**How to Mitigate Position Bias:**

1. **Swap Positions**
   - Evaluate twice: once with "A vs B" and once with "B vs A"
   - Average the results or require consistency
   - Example: If A wins in both positions, it's truly better

2. **Multiple Judge Runs**
   - Run the evaluation multiple times with randomized positions
   - Aggregate results to reduce bias

3. **Explicit Instructions**
   - Add to judge prompt: "Evaluate both responses objectively regardless of their order"
   - Remind judge to consider quality, not position

4. **Statistical Correction**
   - Measure bias on test set with known equal-quality responses
   - Apply correction factor to real evaluations

**Real Impact:**
Without mitigation, position bias can:
- ❌ Make worse models appear better (if systematically placed first)
- ❌ Invalidate A/B test results
- ❌ Create false confidence in evaluation outcomes
- ❌ Lead to wrong decisions about model deployment

**Best Practice:**
Always swap positions and compare results in both orders when using pairwise LLM judges! If results are inconsistent across positions, the comparison is unreliable.

### Verbosity bias

**What is Verbosity Bias?**
Verbosity bias occurs when LLM judges **unfairly favor longer, more detailed responses** over shorter, more concise ones - even when the shorter response is equally good or better. The judge mistakenly equates "more words" with "better quality."

**The Problem:**
- Longer responses tend to win, regardless of actual quality
- Concise, direct answers get penalized
- Judges confuse verbosity with helpfulness or thoroughness
- "Longer = better" even when extra content is irrelevant or repetitive

**Example:**

**Question**: "What is the capital of France?"

**Response A** (Concise): "Paris"

**Response B** (Verbose): "The capital of France is Paris, which is located in the northern central part of the country. Paris is not only the capital but also the largest city in France. It's known for the Eiffel Tower, the Louvre Museum, and its rich cultural history. The city has been the capital since the 12th century and plays a crucial role in French politics, economy, and culture."

**Judge's Rating**: "Response B is better - more comprehensive and informative"

**Problem**: Response A perfectly answers the question! Response B adds unnecessary information.

![alt text](images/verbosity_bias.png)

*This image likely demonstrates verbosity bias through experimental results, showing that longer responses consistently receive higher ratings from LLM judges even when the content quality is equivalent. It may include graphs showing correlation between response length and judge scores, or examples where verbose but lower-quality responses beat concise, high-quality ones.*

**Why Does This Happen?**

1. **Training Data Bias**: LLMs are trained on text where longer explanations are often associated with higher quality content
2. **Surface-Level Signals**: Judges use length as a proxy for effort and detail
3. **Human Preference Mimicry**: Humans sometimes prefer detailed responses, and judges learned this pattern
4. **Lack of Brevity Appreciation**: Judges don't recognize that conciseness can be a virtue

**Real-World Impact:**

- ❌ Penalizes efficient, clear communication
- ❌ Rewards unnecessary verbosity and "fluff"
- ❌ Encourages LLMs to generate longer, potentially less focused responses
- ❌ Misleads optimization - systems get tuned for length, not quality
- ❌ User experience suffers - users often prefer quick, direct answers

**How to Mitigate Verbosity Bias:**

1. **Explicit Instructions**
   - Tell judge: "Prefer concise answers that directly address the question"
   - "Do not favor responses simply because they are longer"
   - "A short, accurate answer is better than a long, unfocused one"

2. **Length-Normalized Scoring**
   - Adjust scores based on response length
   - Penalize unnecessarily long responses
   - Reward efficiency and directness

3. **Task-Specific Criteria**
   - For factual questions: emphasize accuracy over detail
   - For explanations: specify desired level of detail
   - Define "good" explicitly for each task type

4. **Example-Based Guidance**
   - Show judge examples where concise responses are superior
   - Include cases where verbose responses are penalized

5. **Multiple Evaluation Dimensions**
   - Rate "conciseness" as a separate criterion
   - Score "relevance" - does every sentence add value?
   - Evaluate "directness" - does it answer the question quickly?

**Best Practice:**

Always explicitly instruct judges on the expected response length and quality trade-offs. For simple questions, prefer brevity; for complex explanations, detailed responses are appropriate. Context matters!

### Self-enhancement Bias

**What is Self-enhancement Bias?**
Self-enhancement bias (also called **self-preference bias**) occurs when an LLM judge **unfairly favors its own outputs** or outputs from the same model family. It's like a teacher grading their own students higher than others - the judge gives better scores to responses it generated or that match its style.

**The Problem:**
- GPT-4 as judge rates GPT-4 responses higher than Claude responses
- Even when quality is equal, judges prefer their "own" responses
- Creates unfair evaluation when comparing different LLM models
- Undermines objectivity in model comparisons

**Example:**

**Question**: "Explain quantum entanglement"

**Response A** (Generated by GPT-4): "Quantum entanglement is a phenomenon where two particles become correlated such that the state of one instantly affects the other, regardless of distance."

**Response B** (Generated by Claude): "Quantum entanglement occurs when particles interact and become connected, so measuring one particle immediately influences the other's state, even across vast distances."

**Judge** (GPT-4): "Response A is better - clearer and more precise"

**Problem**: Both responses are essentially the same quality! Judge favored GPT-4's output because it matches its own style/vocabulary.

![alt text](images/self_enhancement_bias.png)

*This image likely presents experimental evidence of self-enhancement bias, showing win rates when LLMs judge their own outputs versus other models' outputs. For example, it might show "GPT-4 as judge: GPT-4 responses win 65% vs Claude responses win 35%" even when human evaluators rate them equally. The image probably demonstrates this bias across multiple judge-model combinations.*

**Why Does This Happen?**

1. **Style Recognition**: Judges recognize patterns, phrases, and formatting from their own training
2. **Familiarity Bias**: Outputs matching the judge's style feel "more correct"
3. **Training Distribution**: Judges were trained on similar data to the models being evaluated
4. **Implicit Preference**: Judges may have learned to prefer certain expression styles that they themselves use

**Real-World Impact:**

- ❌ **Unfair model comparisons**: Makes one model appear better than it actually is
- ❌ **Invalid benchmarks**: Self-judged evaluations can't be trusted
- ❌ **Biased leaderboards**: Rankings favor models similar to the judge
- ❌ **Poor decision making**: Companies may choose wrong models based on biased evaluations
- ❌ **Circular evaluation**: If GPT-4 judges GPT-4, it's not objective!

**How to Mitigate Self-enhancement Bias:**

1. **Use Different Judge Models**
   - Don't use GPT-4 to judge only GPT-4 outputs
   - Rotate judges: use multiple different models as judges
   - Average results across different judge models
   - Example: Use GPT-4, Claude, and PaLM as judges, then average

2. **Blind Evaluation**
   - Hide which model generated which response
   - Randomize response order and labels
   - Don't tell judge the source of responses

3. **External Judges**
   - Use models that weren't involved in generating responses
   - Prefer independent third-party judge models
   - Human evaluation as ground truth check

4. **Cross-Validation**
   - Have Model A judge Model B's outputs
   - Have Model B judge Model A's outputs
   - Compare results - high disagreement indicates bias

5. **Multiple Judge Consensus**
   - Use 3+ different judge models
   - Only trust results when multiple judges agree
   - Flag cases where judges strongly disagree

6. **Human Calibration**
   - Compare judge scores to human ratings
   - Measure if specific judge-model combinations show bias
   - Apply correction factors based on known biases

**Best Practice:**

**Never use a model to judge only its own outputs!** Always use multiple diverse judges and cross-validate results. For critical evaluations, include human judgment as the gold standard.

**Detection Test:**
To check for self-enhancement bias:
1. Have Model A judge outputs from Models A, B, and C
2. Have Model B judge the same outputs
3. If Model A consistently rates A's outputs higher than Model B does, bias exists!

### Best Practices

![alt text](images/best_practices.png)

### Dimension to evaluate

When using LLM as a Judge, you need to define **specific criteria or dimensions** to evaluate. Rather than just asking "Is this good?", you should specify exactly what aspects of quality matter for your use case. Different applications require different evaluation dimensions.

![alt text](images/evaluation_dimensions.png)

*This image likely lists common evaluation dimensions for LLM outputs, such as: **Factuality** (truthfulness), **Relevance** (answers the question), **Coherence** (logically consistent), **Fluency** (grammatically correct), **Helpfulness** (useful to user), **Safety** (no harmful content), and **Creativity** (novel/interesting). Each dimension measures a different aspect of quality.*

#### Factuality

**What is Factuality?**
Factuality measures whether the LLM's response is **factually correct and truthful**. It checks if the information provided is accurate, based on real facts, and not made up (hallucinated). This is one of the most critical dimensions for applications where accuracy matters.

**Why It Matters:**
- LLMs can confidently state false information (hallucinations)
- Users trust LLM outputs and may act on incorrect information
- Factual errors can have serious consequences (medical, legal, financial domains)
- Critical for knowledge-based applications

**What to Check:**
- ✓ Are facts verifiable and correct?
- ✓ Are dates, numbers, names accurate?
- ✓ Is the information up-to-date?
- ✓ Are claims supported by evidence?
- ✗ Does the response contain hallucinations?
- ✗ Are there fabricated details?

![alt text](images/factuality_examples.png)

*This image likely shows examples of factual vs. non-factual responses. For instance, a question like "When did World War 2 end?" with correct answer "1945" (factual) vs incorrect answer "1950" (non-factual). It may also demonstrate hallucinations where an LLM invents plausible-sounding but false information.*

![alt text](images/factuality_evaluation.png)

*This image probably illustrates methods for evaluating factuality, such as: **Knowledge-based verification** (checking against trusted sources), **Fact-checking tools** (automated verification systems), **Citation checking** (verifying sources cited), **Claim extraction and validation** (breaking response into individual claims and verifying each), or **LLM judge prompts** specifically designed to detect factual errors and hallucinations.*

![alt text](images/factuality_scoring.png)

*This image likely shows a practical example or rubric for scoring factuality. It might include a scoring scale (e.g., 1-5 where 1=Many errors, 3=Mostly accurate with minor errors, 5=Completely accurate) and sample judge outputs evaluating factuality with explanations like "Score: 2/5 - Response incorrectly states Paris is the capital of Italy (should be France) and claims the Eiffel Tower was built in 1920 (actually 1889)."*

**How to Evaluate Factuality with LLM Judge:**

**Example Judge Prompt:**
```
Evaluate the factual accuracy of the response:
Question: [question]
Response: [response]

Check for:
1. Factual errors (incorrect information)
2. Hallucinations (made-up information)
3. Outdated information
4. Unsupported claims

Rate factuality on scale of 1-5:
1 = Multiple major factual errors
2 = Some factual errors present
3 = Mostly accurate with minor issues
4 = Accurate with very minor issues
5 = Completely factually accurate

Provide your rating and explain any errors found.
```

**Real Example:**

**Question**: "What is the capital of Australia?"

**Response A**: "The capital of Australia is Sydney."
- **Factuality Score**: 1/5 - **Incorrect!** Capital is Canberra, not Sydney

**Response B**: "The capital of Australia is Canberra."
- **Factuality Score**: 5/5 - **Correct!**

**Response C**: "The capital of Australia is Canberra, established in 1927 to resolve the rivalry between Sydney and Melbourne."
- **Factuality Score**: 5/5 - **Correct and includes accurate historical context**

**Challenges in Evaluating Factuality:**

1. **Judge Knowledge Limitations**: Judge LLM might not know all facts
2. **Outdated Training Data**: Judge may have outdated information
3. **Subjective Facts**: Some claims are debatable, not clearly true/false
4. **Missing Context**: Without external knowledge, judge can't verify everything
5. **Confident Hallucinations**: Both generator and judge might hallucinate the same false facts

**Best Practices:**

- **Use external verification**: Don't rely on judge LLM alone - use fact-checking tools, search APIs, or knowledge bases
- **Provide ground truth**: When possible, give judge access to verified sources
- **Break down claims**: Evaluate individual factual claims separately
- **Focus on verifiable facts**: Names, dates, numbers, events that can be checked
- **Human verification for critical domains**: Always have human review for high-stakes applications (medical, legal, financial)

---

# Agent Evaluation

**What is Agent Evaluation?**
Unlike simple chatbots, LLM agents can use tools, plan multi-step actions, and interact with external systems. Agent evaluation measures how well they complete complex tasks, not just how good their text responses are.

**Key Difference**: Traditional LLM evaluation focuses on text quality. Agent evaluation focuses on task success—did the agent actually achieve the goal using the right tools and steps?

![alt text](images/agent_evaluation_workflow.png)

*This image likely shows the agent workflow: receiving a complex task (e.g., "Book a flight to Paris"), breaking it into steps (search flights, compare prices, book), using various tools at each step, and evaluation checkpoints measuring whether it chose correct tools and completed the task successfully.*

![alt text](images/agent_evaluation_metrics.png)

*This image probably displays key agent metrics: **Task Success Rate** (did it complete the goal?), **Action Accuracy** (were individual steps correct?), **Tool Selection** (right tools chosen?), **Efficiency** (number of steps), and **Cost** (API calls/tokens used). It may compare successful vs. failed agent executions.*

## Tool Prediction Error

**What is Tool Prediction Error?**
This occurs when an agent selects the wrong tool or uses incorrect parameters. For example, calling `search_web()` instead of `calculator()` for math, or using wrong date formats in function calls.

**Why It Matters**: Wrong tool choice = task failure. This is one of the most common reasons agents fail in real-world use.

![alt text](images/tool_prediction_error.png)

*This image likely demonstrates tool prediction errors with examples: user asks "What's 25% of 150?", agent incorrectly calls `search_web("25% of 150")` instead of `calculator(0.25 * 150)`. It probably shows error types: wrong tool selection, missing parameters, incorrect parameter types, and invalid values.*

## Hallucination Tool

**What are Hallucination Tools?**
Agents sometimes "hallucinate" by inventing tools that don't exist, making up parameters, or creating fake outputs. Hallucination detection tools validate that agent actions match the actual available APIs and tools.

![alt text](images/hallucination_tool.png)

*This image likely shows agent hallucination examples: calling non-existent tools (e.g., `get_live_stock_data()` when it doesn't exist), fabricating parameters on real tools, or inventing plausible-sounding but false outputs. It may demonstrate detection methods like checking against actual API schemas.*

## Using wrong tool

**What is "Using Wrong Tool"?**
This error occurs when an agent selects an inappropriate tool for the task. For example, using a search engine to calculate math instead of a calculator, or calling an email function when the user asked for a web search.

**Example**: User asks "What's 15 × 8?" → Agent calls `web_search("15 times 8")` instead of `calculator(15 * 8)` ❌

![alt text](images/using_wrong_tool.png)

*This image likely demonstrates real examples of wrong tool selection: showing scenarios where agents choose inappropriate tools (e.g., using text generation for calculations, using database query for web searches, or using translation tool for summarization). It probably highlights the mismatch between user intent and tool capability.*

## Infers wrong arguments

**What is "Infers Wrong Arguments"?**
This error occurs when an agent selects the correct tool but provides incorrect parameters or arguments. The tool choice is right, but the values, types, or formats passed to the tool are wrong.

**Example**: Agent correctly calls `get_weather()` but uses `get_weather(city="USA")` instead of a specific city like `get_weather(city="New York")` ❌

![alt text](images/infers_wrong_arguments.png)

*This image likely shows examples of parameter errors: agent calls the right tool but with wrong argument types (string instead of number), wrong formats (date as "tomorrow" instead of "2024-12-26"), missing required parameters, or invalid values. It probably demonstrates how these errors cause tool execution failures.*

## Wrong Response

**What is "Wrong Response"?**
This error occurs when an agent executes actions but produces an incorrect or inappropriate final response. The agent may have used tools correctly, but fails to interpret results properly or generates a response that doesn't match the user's needs.

**Example**: User asks "What's the weather in Tokyo?" → Agent calls `get_weather(city="Tokyo")` correctly, gets "22°C", but responds "It's very cold in Tokyo" ❌

![alt text](images/wrong_response.png)

*This image likely demonstrates examples of wrong responses: agent retrieves correct data but misinterprets it, provides irrelevant information, formats output incorrectly, or draws wrong conclusions from tool results. It may show cases where tool execution succeeded but the final answer is still wrong.*

## No Response

**What is "No Response"?**
This error occurs when an agent fails to provide any response at all. The agent may get stuck, encounter an error it can't recover from, or fail to complete its reasoning process.

**Example**: User asks "Book a flight to Paris" → Agent starts processing but never returns an answer (timeout, infinite loop, or crash) ❌

![alt text](images/no_response_scenarios.png)

*This image likely shows scenarios causing no response: agent encountering unrecoverable errors, getting stuck in reasoning loops, timing out during tool execution, or failing to handle edge cases. It may illustrate common failure patterns and their causes.*

![alt text](images/no_response_impact.png)

*This image probably demonstrates additional no-response scenarios or shows the impact on user experience: users waiting indefinitely, system timeouts, or strategies to detect and handle agent failures (timeout mechanisms, fallback responses, error monitoring).*

![alt text](images/agent_error_summary.png)

*This image likely summarizes agent error types or provides best practices for handling agent failures: implementing timeout mechanisms, adding error recovery strategies, monitoring agent health, providing fallback responses, and graceful degradation when agents fail.*

---

# Benchmarks

**What are Benchmarks?**
Benchmarks are standardized test sets used to evaluate and compare LLM performance across different models. They provide consistent, objective measurements of capabilities like reasoning, knowledge, math, coding, and language understanding.

**Why Benchmarks Matter:**
- **Compare models**: See which LLM performs best on specific tasks
- **Track progress**: Measure improvements over time
- **Identify weaknesses**: Find areas where models struggle
- **Standardized testing**: Everyone uses same tests = fair comparison

## Common Benchmarks

![alt text](images/common_benchmarks.png)

*This image likely lists popular LLM benchmarks such as: **MMLU** (knowledge across subjects), **HumanEval** (coding), **GSM8K** (math word problems), **HellaSwag** (commonsense reasoning), **TruthfulQA** (factual accuracy), **BBH** (Big Bench Hard - challenging tasks), and **ARC** (science questions). It may show what each benchmark tests and typical score ranges.*

## MMLU (Massive Multitask Language Understanding) - Knowledge

**What is MMLU?**
MMLU is one of the most widely used benchmarks for testing LLM knowledge across **57 different subjects** including math, history, law, medicine, computer science, and more. It contains multiple-choice questions that test both breadth and depth of knowledge.

**Why It's Important:**
- Tests general knowledge across diverse domains
- Questions range from elementary to expert level
- Widely used to compare model capabilities
- Score represents percentage of correct answers (0-100%)

**Example Questions:**
- Math: "What is the derivative of x²?"
- History: "When did World War I begin?"
- Medicine: "What is the primary function of mitochondria?"

![alt text](images/mmlu_structure.png)

*This image likely shows MMLU structure: the 57 subject categories (STEM, humanities, social sciences, etc.), difficulty levels, question format (multiple choice with 4 options), and example questions from different domains. It may include a breakdown of how questions are distributed across subjects.*

![alt text](images/mmlu_performance.png)

*This image probably displays MMLU performance comparisons: showing scores for different LLMs (GPT-4, Claude, PaLM, etc.) across categories, highlighting which models excel in specific subjects, and demonstrating score improvements over model generations. It may show that top models achieve 85-90% accuracy while older models score 50-60%.*

---

## Reasoning

**What is Reasoning Evaluation?**
Reasoning benchmarks test an LLM's ability to think logically, solve complex problems, and apply multi-step thinking. Unlike knowledge tests (memorized facts), reasoning tests measure whether models can actually think through problems, use logic, and reach correct conclusions.

**Why It Matters:**
- Tests problem-solving ability, not just memorization
- Evaluates logical thinking and inference skills
- Critical for applications requiring analysis and decision-making
- Shows whether models can handle novel problems

### AIME (Math)

AIME = `American Invitational Mathematics Examination`

**What is AIME?**
AIME is a challenging high school mathematics competition used to evaluate LLM mathematical reasoning. Problems require multi-step solutions, creative thinking, and deep understanding of mathematical concepts—not just formula memorization.

**Why It's Important:**
- Tests advanced mathematical reasoning
- Problems require 5-10 steps to solve
- Evaluates logical thinking and problem decomposition
- Much harder than basic arithmetic benchmarks

![alt text](images/aime_problems.png)

*This image likely shows example AIME problems: complex math questions requiring algebra, geometry, number theory, or combinatorics. Problems might include finding specific values, proving relationships, or calculating probabilities. It probably demonstrates the difficulty level and multi-step nature of AIME questions.*

![alt text](images/aime_performance.png)

*This image probably displays AIME performance results: showing accuracy scores for different LLMs (e.g., GPT-4 solves 15-20% of AIME problems, while older models score near 0%). It may illustrate the gap between top models and average models, highlighting that AIME is challenging even for advanced LLMs.*

### PIQA

**What is PIQA?**
PIQA (Physical Interaction Question Answering) tests **commonsense physical reasoning** about everyday situations. It asks which action would achieve a goal in the real world, testing whether LLMs understand basic physics and practical knowledge.

**Example Question:**
"To separate egg whites from the yolk..."
- A) "crack the egg over a bowl and use your hands to catch the yolk"
- B) "crack the egg into a bottle and squeeze" ✓

**Why It Matters:**
- Tests practical, real-world reasoning
- Measures commonsense understanding
- Evaluates if models grasp physical cause-and-effect
- Important for applications involving real-world tasks

![alt text](images/piqa_examples.png)

*This image likely shows PIQA question examples: everyday scenarios with two possible solutions where one is clearly more practical or effective. Questions might cover cooking, repairs, cleaning, or other physical tasks requiring commonsense reasoning about how objects and materials behave.*

![alt text](images/piqa_results.png)

*This image probably displays PIQA benchmark results: showing accuracy scores for different LLMs (top models achieve 85-95% accuracy). It may compare model performance, demonstrating that most modern LLMs handle commonsense physical reasoning well, though older models struggled more.*

---

## Coding

**What is Coding Evaluation?**
Coding benchmarks test an LLM's ability to write, understand, and debug programming code. These tests measure whether models can generate correct, functional code that solves specific problems or implements features.

**Why It Matters:**
- Evaluates practical programming ability
- Tests understanding of syntax, algorithms, and logic
- Critical for AI coding assistants and development tools
- Measures ability to translate requirements into working code

### SWE

**What is SWE Bench?**
SWE-bench (Software Engineering Benchmark) tests LLMs on **real-world software engineering tasks** from actual GitHub repositories. Models must understand existing codebases, identify bugs, and implement fixes or features—not just write isolated code snippets.

**Why It's Important:**
- Tests real-world coding skills (not toy problems)
- Requires understanding large, complex codebases
- Evaluates ability to fix actual bugs from open-source projects
- Much harder than simple "write a function" tasks
- Reflects realistic developer workflows

![alt text](images/swe_bench_tasks.png)

*This image likely shows SWE-bench task examples: real GitHub issues from popular repositories (like Django, Flask, scikit-learn) where LLMs must read issue descriptions, understand the codebase, locate bugs, and generate correct patches or implementations. It probably demonstrates the complexity of real-world software engineering tasks.*

![alt text](images/swe_bench_results.png)

*This image probably displays SWE-bench performance results: showing success rates for different LLMs (e.g., top models solve 10-30% of issues, demonstrating how challenging real-world code tasks are). It may compare models and show that even advanced LLMs struggle with complex codebases, highlighting the gap between simple coding tests and actual software engineering.*

---

## Safety

**What is Safety Evaluation?**
Safety benchmarks test whether LLMs can be manipulated to generate harmful, dangerous, or unethical content. These tests measure how well models resist malicious prompts and refuse inappropriate requests while remaining helpful for legitimate use cases.

**Why It Matters:**
- Prevents misuse of AI systems
- Protects users from harmful content
- Tests model alignment with ethical guidelines
- Critical for deploying LLMs in production
- Required for regulatory compliance

### HarmBench

**What is HarmBench?**
HarmBench evaluates LLM safety by testing resistance to **adversarial attacks** designed to bypass safety guardrails. It uses sophisticated prompt injection techniques, jailbreaks, and manipulation strategies to see if models can be tricked into generating harmful content.

**Why It's Important:**
- Tests real-world attack scenarios (not just obvious harmful requests)
- Measures robustness of safety mechanisms
- Identifies vulnerabilities in model alignment
- Helps improve safety before deployment
- Covers diverse harm categories (violence, illegal activities, misinformation, etc.)

**What It Tests:**
- Resistance to jailbreak attempts
- Handling of harmful requests
- Consistency of safety refusals
- Ability to maintain safety under adversarial conditions

![alt text](images/harmbench_attacks.png)

*This image likely shows HarmBench attack examples: sophisticated prompts trying to manipulate LLMs into generating harmful content using techniques like role-playing scenarios, encoded requests, multi-step manipulations, or context confusion. It probably demonstrates various jailbreak strategies that researchers use to test safety boundaries.*

![alt text](images/harmbench_results.png)

*This image probably displays HarmBench results: showing attack success rates (ASR) for different LLMs, where lower scores are better (e.g., model refuses harmful requests 95% of the time). It may compare safety across models, demonstrating which LLMs have stronger safety guardrails and which are more vulnerable to manipulation attempts.*

---

## Agent Benchmarking

**What is Agent Benchmarking?**
Agent benchmarks evaluate LLM-powered agents on their ability to complete complex, multi-step tasks using tools and interacting with real-world systems. Unlike simple text generation tests, these benchmarks measure task completion, tool usage accuracy, reasoning quality, and goal achievement.

**Why It Matters:**
- Tests real-world agent capabilities
- Evaluates tool selection and execution
- Measures multi-step planning and reasoning
- Critical for deploying agents in production
- Goes beyond text quality to actual task success

### Tau-Bench(𝜏-bench)

`𝜏-bench = Tool-Agent-User Interaction Benchmark`

**What is Tau-Bench?**
Tau-bench (𝜏-bench) evaluates agents on realistic tasks requiring **tool use, user interaction, and multi-turn dialogue**. It tests whether agents can understand user intent, select appropriate tools, execute actions correctly, and adapt based on feedback—simulating real assistant scenarios.

**Why It's Important:**
- Tests realistic user-agent interactions
- Evaluates multi-turn conversations (not single queries)
- Measures tool usage in context
- Tests ability to handle ambiguity and clarifications
- Reflects real-world assistant deployment scenarios

**What It Tests:**
- Tool selection accuracy
- Parameter inference from conversation
- Handling follow-up questions
- Adapting to user corrections
- Task completion across multiple turns

![alt text](images/tau_bench_tasks.png)

*This image likely shows Tau-bench task examples: realistic scenarios like "Help me book a restaurant reservation" or "Find and summarize recent news on AI" where agents must engage in multi-turn dialogue, ask clarifying questions, use appropriate tools (search, reservations, calendar), and complete tasks based on user feedback.*

![alt text](images/tau_bench_framework.png)

*This image probably displays the Tau-bench evaluation framework: showing how tasks are structured with user goals, available tools, conversation turns, and success criteria. It may illustrate the evaluation process measuring tool accuracy, conversation quality, task completion rate, and efficiency across multiple interaction rounds.*

![alt text](images/tau_bench_performance.png)

*This image likely presents Tau-bench performance results: comparing different LLM agents (GPT-4, Claude, etc.) on metrics like task success rate, tool selection accuracy, number of turns needed, and user satisfaction. It probably demonstrates which agents handle realistic interactions best and highlights common failure modes in multi-turn tool-using scenarios.*

`Gemini 3 Benchmark`

![alt text](images/gemini_benchmark_1.png)

![alt text](images/gemini_benchmark_2.png)

![alt text](images/gemini_benchmark_3.png)

## Pareto frontier

**What is Pareto Frontier?**
The Pareto frontier (or Pareto optimal curve) shows the **best trade-offs between competing metrics** when evaluating LLMs. It helps identify models that offer the best balance between factors like performance vs. cost, accuracy vs. speed, or quality vs. inference time.

**Why It Matters:**
- No single model is perfect at everything
- Helps choose the right model for your constraints
- Shows which models are truly better vs. just different
- Visualizes trade-offs between metrics (e.g., accuracy vs. cost)

**Key Concept:**
- **On the frontier**: Models that are optimal - you can't improve one metric without sacrificing another
- **Below the frontier**: Suboptimal models - there exists a better option
- **Example**: Model A (95% accuracy, $10/M tokens) vs. Model B (90% accuracy, $1/M tokens) - both on frontier, choose based on your priority

![alt text](images/pareto_frontier_concept.png)

*This image likely shows a Pareto frontier plot with two competing metrics (e.g., performance on y-axis vs. cost on x-axis). Points on the upper-left curve represent optimal models - each offers the best performance for a given cost. Models below the curve are suboptimal. The plot helps visualize which models dominate and where trade-offs exist.*

![alt text](images/pareto_frontier_llm_comparison.png)

*This image probably displays a real-world Pareto frontier comparing LLMs across multiple dimensions (e.g., MMLU score vs. inference cost, or accuracy vs. latency). It may show where popular models like GPT-4, Claude, Llama, and Gemini fall on the curve, demonstrating which models are Pareto optimal and helping users choose based on their specific constraints (budget, speed requirements, accuracy needs).*

## Data contamination

![alt text](images/data_contamination_1.png)

![alt text](images/data_contamination_2.png)

![alt text](images/data_contamination_3.png)