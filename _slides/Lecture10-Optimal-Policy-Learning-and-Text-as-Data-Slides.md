---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 10 | Optimal Policy Learning & Text-as-Data'
size: 16:9
style: |
  @import 'default';

  section {
    background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
    font-family: 'Segoe UI', 'Liberation Sans', sans-serif;
    font-size: 22px;
    padding: 70px 80px;
    color: #1e293b;
    line-height: 1.8;
  }

  section::after {
    font-size: 12px;
    color: #64748b;
  }

  h1 {
    color: #0f172a;
    font-weight: 700;
    font-size: 2em;
    margin-bottom: 0.5em;
    border-bottom: 4px solid #5a3bf6;
    padding-bottom: 0.3em;
  }

  h2 {
    color: #361eaf;
    font-weight: 600;
    font-size: 1.5em;
    margin-top: 0.8em;
    margin-bottom: 0.6em;
  }

  h3 {
    color: #3730a3;
    font-weight: 600;
    font-size: 1.2em;
  }

  strong {
    color: #0f172a;
    font-weight: 600;
  }

  ul, ol {
    margin-left: 1.5em;
    line-height: 2.2;
  }

  li {
    margin-bottom: 0.8em;
  }

  p {
    line-height: 1.9;
  }

  code {
    background: #e0e7ff;
    color: #3730a3;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
  }

  pre {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    font-size: 0.85em;
  }

  pre code {
    background: transparent;
    color: #e2e8f0;
    padding: 0;
  }

  table {
    margin: 25px auto;
    border-collapse: collapse;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-size: 0.95em;
  }

  table th {
    background: linear-gradient(to bottom, #3b82f6, #4325eb);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
  }

  table td {
    padding: 8px 16px;
    border-bottom: 1px solid #e2e8f0;
  }

  table tr:nth-child(even) {
    background: #f8fafc;
  }

  table tr:hover {
    background: #eff6ff;
  }

  blockquote {
    border-left: 4px solid #3b82f6;
    padding-left: 20px;
    margin: 20px 0;
    font-style: italic;
    color: #475569;
  }

  a {
    color: #2563eb;
    text-decoration: none;
    border-bottom: 1px solid #93c5fd;
  }

  a:hover {
    color: #1d4ed8;
    border-bottom-color: #2563eb;
  }

  section.lead {
    background: linear-gradient(135deg, #1e40af 0%, #6d3bf6 100%);
    color: white;
    text-align: center;
    justify-content: center;
  }

  section.lead h1 {
    color: white;
    border-bottom: 4px solid rgba(255,255,255,0.3);
    font-size: 2.2em;
  }

  section.lead h2 {
    color: #dbeafe;
    font-size: 1.4em;
  }

  section.lead strong {
    color: #fbbf24;
  }
---

<!-- _class: lead -->

# Lecture 10
## Optimal Policy Learning & Text-as-Data

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**Part 1: Optimal Policy Learning**
- From inference to decision-making
- Empirical Welfare Maximization (EWM)
- Policy trees and doubly robust scoring

**Part 2: Text as Data**
- Text representation methods
- Sentiment analysis and topic modeling
- Causal inference with text data

---

<!-- _class: lead -->

# Part 1
## Optimal Policy Learning

---

## From Estimation to Decision

**The journey so far:**
- Prediction: Machine learning for $\hat{y}$
- Identification: DAGs, CIA, IV for causal effects
- Estimation: DML, Causal Forests for $\hat{\tau}(x)$

**The missing piece: Decision-making**
- How do we use estimated effects to make policy decisions?
- With budget constraints, who should receive treatment?

---

## The Policy Problem: Motivation

**Standard treatment effect estimation:**
- We estimate heterogeneous effects $\tau(x) = E[Y(1) - Y(0) | X = x]$
- But knowing $\tau(x)$ is not enough for policy

**Real-world constraints:**
- Limited budget (can't treat everyone)
- Treatment has costs $c > 0$
- Some individuals benefit more than others

---

## Example: Job Training Program

**Setting:**
- Government has budget to train 1,000 workers
- 10,000 unemployed workers available
- Training cost: $5,000 per person
- Estimated wage gains vary: $\tau(x_i) \in [-2000, 15000]$

**Key question:**
- Who should receive training to maximize welfare?
- Not just "what is the average effect?"

---

## The Policy Learning Framework

**Formal setup:**
- Policy rule: $\pi: \mathcal{X} \to \{0, 1\}$
- For individual with features $X$, assign treatment $\pi(X)$

**Objective: Maximize expected welfare**
$$V(\pi) = E[Y(\pi(X))]$$

**With treatment cost $c$:**
$$V(\pi) = E[\pi(X) \cdot (Y(1) - c) + (1-\pi(X)) \cdot Y(0)]$$

---

## Optimal Policy Under Known Effects

**If we knew $\tau(x)$ and $c$ exactly:**

$$\pi^*(x) = \begin{cases}
1 & \text{if } \tau(x) > c \\
0 & \text{if } \tau(x) \leq c
\end{cases}$$

**Intuition:**
- Treat individual $i$ if benefit $\tau(x_i)$ exceeds cost $c$
- Simple threshold rule

---

## The Challenge: Estimation Uncertainty

**In practice, we only have $\hat{\tau}(x)$:**
- Estimated from finite sample
- Contains statistical uncertainty
- May be biased or imprecise

**Naive approach fails:**
- Using $\pi(x) = \mathbb{1}\{\hat{\tau}(x) > c\}$ can be suboptimal
- Ignores estimation error
- May perform poorly out-of-sample

---

## Empirical Welfare Maximization (EWM)

<!-- Suggested image: Welfare comparison plot showing different policies' expected outcomes -->

**Key idea (Manski 2004, Kitagawa & Tetenov 2018):**
- Directly optimize expected welfare on sample
- Choose policy $\pi$ from class $\Pi$

**Objective:**
$$\hat{\pi} = \arg\max_{\pi \in \Pi} \frac{1}{n} \sum_{i=1}^n \left[\pi(X_i) Y_i(1) + (1-\pi(X_i)) Y_i(0)\right]$$

**Problem: We don't observe both $Y_i(1)$ and $Y_i(0)$!**

---

## The Fundamental Problem Revisited

**What we observe:**
$$Y_i^{obs} = W_i Y_i(1) + (1-W_i) Y_i(0)$$

**What we need for welfare evaluation:**
$$E[Y(\pi(X))] = E[\pi(X) Y(1) + (1-\pi(X)) Y(0)]$$

**Solution: Use counterfactual prediction methods**
- Imputation via ML
- Inverse propensity weighting
- Doubly robust methods

---

## Athey & Wager (2021): Policy Learning

**Breakthrough approach (Econometrica 2021):**
- Transform policy learning into weighted classification
- Use doubly robust scores as individual-level welfare weights

**Framework:**
$$\hat{\pi} = \arg\max_{\pi \in \Pi} \frac{1}{n} \sum_{i=1}^n \hat{\Gamma}_i \cdot \pi(X_i)$$

where $\hat{\Gamma}_i$ is the doubly robust welfare score

---

## Doubly Robust Welfare Scores

**For individual $i$, define:**

$$\hat{\Gamma}_i = \hat{\tau}(X_i) + \frac{W_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}$$

**Components:**
- $\hat{\tau}(X_i)$: predicted treatment effect
- $\hat{\mu}_1(X_i)$: predicted outcome under treatment
- $\hat{\mu}_0(X_i)$: predicted outcome under control
- $\hat{e}(X_i)$: propensity score

---

## Why Doubly Robust Scores?

**Key properties:**
1. **Unbiased** for true welfare if either $\hat{\mu}$ or $\hat{e}$ is correct
2. **Efficient**: achieves semiparametric efficiency bound
3. **Works with observational data**: accounts for confounding

**Intuition:**
- First term: direct prediction of benefit
- Second/third terms: bias correction using observed outcomes
- Combines imputation and weighting

---

## Numerical Example: DR Scores

**Individual with $X_i = x$:**
- Observed: $W_i = 1$, $Y_i = 15$
- Predictions: $\hat{\tau}(x) = 5$, $\hat{\mu}_1(x) = 12$, $\hat{e}(x) = 0.5$

**DR score calculation:**
$$\hat{\Gamma}_i = 5 + \frac{1 \times (15 - 12)}{0.5} - 0 = 5 + 6 = 11$$

**Interpretation:** Treating this individual yields welfare gain of 11

---

## Policy Learning as Classification

**Reformulation:**
- Each individual $i$ has features $X_i$ and weight $\hat{\Gamma}_i$
- Goal: classify individuals into treatment/control
- Maximize: $\sum_{i=1}^n \hat{\Gamma}_i \cdot \pi(X_i)$

**This is weighted classification!**
- Standard ML algorithms apply
- Can use trees, forests, neural nets
- Policy class $\Pi$ determines complexity

---

## Policy Trees

<!-- Suggested image: Policy tree visualization showing decision rules for treatment allocation -->

**Simple interpretable policies:**
- Partition feature space into regions
- Assign uniform treatment within each region

**Example: Job training**
```
if Age < 30:
    if Education >= 12:
        Treat
    else:
        Don't treat
else:
    Don't treat
```

---

## Building Policy Trees: Algorithm

**Greedy algorithm (similar to CART):**

1. Start with all data in root node
2. For each possible split $(j, s)$ on feature $X_j$ at value $s$:
   - Compute welfare in left/right child
   - Choose treatment for each child
3. Select split maximizing total welfare gain
4. Recurse until stopping criterion

**Key difference from regression trees:** maximize welfare, not reduce variance

---

## Policy Tree: Splitting Criterion

**For candidate split creating children $L, R$:**

$$\text{Welfare Gain} = \max_{d_L, d_R \in \{0,1\}} \left[\sum_{i \in L} \hat{\Gamma}_i d_L + \sum_{i \in R} \hat{\Gamma}_i d_R\right]$$

**Optimal treatment assignment:**
- $d_L = \mathbb{1}\{\sum_{i \in L} \hat{\Gamma}_i > 0\}$
- $d_R = \mathbb{1}\{\sum_{i \in R} \hat{\Gamma}_i > 0\}$

**Intuition:** Treat if average welfare score in region is positive

---

## Numerical Example: Policy Tree Split

**Node with 4 individuals:**

| $i$ | $X_i$ (Age) | $\hat{\Gamma}_i$ |
|-----|-------------|------------------|
| 1   | 25          | 8                |
| 2   | 28          | 6                |
| 3   | 35          | -3               |
| 4   | 40          | -5               |

**Split at Age = 30:**
- Left $(i=1,2)$: $\sum \hat{\Gamma}_i = 14 > 0$ → Treat
- Right $(i=3,4)$: $\sum \hat{\Gamma}_i = -8 < 0$ → Don't treat
- Total welfare: $14 + 0 = 14$

---

## Budget Constraints

**With fixed budget $B$ (can treat at most $k$ individuals):**

**Modified objective:**
$$\max_{\pi} \sum_{i=1}^n \hat{\Gamma}_i \pi(X_i) \quad \text{s.t.} \quad \sum_{i=1}^n \pi(X_i) \leq k$$

**Solution: Top-k rule**
- Rank individuals by $\hat{\Gamma}_i$
- Treat top $k$ individuals

---

## Policy Learning with Costs

**With heterogeneous treatment costs $c(X)$:**

**Net benefit score:**
$$\hat{\Gamma}_i^{net} = \hat{\Gamma}_i - c(X_i)$$

**Optimal policy:**
$$\pi^*(X_i) = \mathbb{1}\{\hat{\Gamma}_i^{net} > 0\}$$

**Example:** Training costs vary by location, skill requirements

---

## Cross-Fitting for Policy Learning

**Avoid overfitting:**
1. Split sample into $K$ folds
2. For fold $k$:
   - Estimate $\hat{\tau}, \hat{\mu}_1, \hat{\mu}_0, \hat{e}$ on other folds
   - Compute $\hat{\Gamma}_i$ for individuals in fold $k$
3. Learn policy $\hat{\pi}$ on all DR scores

**Ensures valid statistical inference**

---

## Evaluating Policy Performance

**Challenge:** How good is our learned policy?

**Evaluation metric: Value function**
$$V(\pi) = E[Y(\pi(X))]$$

**Estimation via inverse propensity weighting:**
$$\hat{V}(\pi) = \frac{1}{n} \sum_{i=1}^n \frac{\mathbb{1}\{W_i = \pi(X_i)\} Y_i}{\hat{e}(X_i)^{W_i}(1-\hat{e}(X_i))^{1-W_i}}$$

---

## Policy Regret

**Compare learned policy to oracle:**
$$\text{Regret}(\hat{\pi}) = V(\pi^*) - V(\hat{\pi})$$

where $\pi^*(x) = \mathbb{1}\{\tau(x) > c\}$

**Theoretical guarantees:**
- Under regularity conditions, regret $\to 0$ as $n \to \infty$
- Rate depends on flexibility of policy class $\Pi$

---

## Empirical Application: Medicare Part D

**Abaluck et al. (2020):**
- Helping seniors choose prescription drug plans
- Many plans available, choices are suboptimal
- Goal: recommend personalized plan

**Policy learning approach:**
- Features: age, health conditions, medications
- Outcome: out-of-pocket costs
- Learn policy to minimize costs

---

## Results: Medicare Part D

**Findings:**
- Simple policy trees achieve 90% of optimal savings
- Average savings: $300 per person per year
- Interpretable rules (e.g., "if diabetic and 3+ meds, choose plan A")

**Implementation:**
- Deployed as decision support tool
- Demonstrates real-world value of policy learning

---

## Comparison: Methods for Policy Learning

| Method | Interpretability | Performance | Constraints |
|--------|-----------------|-------------|-------------|
| Threshold on $\hat{\tau}$ | High | Good | None |
| Policy Trees | High | Better | Budget, costs |
| Policy Forests | Medium | Best | Budget, costs |
| Neural Nets | Low | Best | Any |

---

## Software: Policy Learning in Practice

**R packages:**
- `policytree`: implements policy trees with doubly robust scores
- `grf`: includes policy learning via generalized random forests

**Python:**
- `EconML`: Microsoft's package with policy learning modules
- `CausalML`: Uber's package with uplift modeling

---

## Key Takeaways: Optimal Policy Learning

**Main insights:**
- Policy learning bridges causal inference and decision-making
- Doubly robust scores enable welfare maximization
- Can handle budget constraints and heterogeneous costs
- Trade-off between interpretability and performance

**When to use:**
- Limited resources require targeting
- Treatment effects are heterogeneous
- Need interpretable or implementable rules

---

<!-- _class: lead -->

# Part 2
## Text as Data

---

## Why Economists Care About Text

**Traditional economic data:**
- Prices, quantities, income, GDP
- Well-structured, numerical

**The information revolution:**
- News articles, social media, financial reports
- Central bank communications, policy documents
- Product reviews, job postings

**Text contains rich economic information!**

---

## Applications of Text in Economics

**Monetary policy:**
- Measuring hawkish/dovish stance from FOMC minutes
- Central bank communication and market reactions

**Finance:**
- Sentiment in earnings calls and stock returns
- Credit risk from firm disclosures

**Labor:**
- Job posting requirements and wage inequality
- Skills demanded across occupations

---

## The Fundamental Challenge

**Computers don't understand language:**
- "Good" is just a string of characters
- Need to convert text to numbers
- Preserve semantic meaning

**The representation problem:**
- How to map text to vectors?
- Billions of possible sentences
- Complex linguistic structures

---

## Text Representation: Overview

**Historical progression:**
1. Bag of Words (BoW): word counts
2. TF-IDF: weighted word importance
3. Word Embeddings: semantic vectors (Word2Vec, GloVe)
4. Contextual Embeddings: context-aware (BERT, GPT)

**Trade-offs:**
- Simplicity vs. expressiveness
- Interpretability vs. performance
- Computational cost

---

## Bag of Words (BoW)

<!-- Suggested image: Word cloud from economic text showing frequent terms -->

**Simplest representation:**
- Count word frequencies in document
- Ignore word order, grammar, context

**Example documents:**
- Doc 1: "inflation is rising"
- Doc 2: "rising inflation concerns"

**Vocabulary:** {inflation, is, rising, concerns}

---

## BoW: Numerical Example

**Document-term matrix:**

|     | inflation | is | rising | concerns |
|-----|-----------|----|---------|---------:|
| Doc 1 | 1       | 1  | 1      | 0        |
| Doc 2 | 1       | 0  | 1      | 1        |

**Vector representation:**
- Doc 1: $[1, 1, 1, 0]$
- Doc 2: $[1, 0, 1, 1]$

---

## BoW: Strengths and Weaknesses

**Strengths:**
- Simple, interpretable
- Works well for topic classification
- Computationally efficient

**Weaknesses:**
- Loses word order ("dog bites man" vs. "man bites dog")
- Ignores semantics ("good" vs. "great")
- High dimensionality (vocabulary size)
- Sparse vectors

---

## TF-IDF: Term Frequency-Inverse Document Frequency

**Motivation:**
- "The" appears frequently but carries little information
- "Inflation" is more informative

**Weighting scheme:**
$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$$

where:
- $\text{TF}(w, d)$: frequency of word $w$ in document $d$
- $\text{IDF}(w) = \log \frac{N}{n_w}$: $N$ = total docs, $n_w$ = docs containing $w$

---

## TF-IDF: Numerical Example

**Corpus: 100 FOMC statements**
- Word "inflation" appears 80 times in current statement
- Appears in 95 out of 100 statements
- Word "cryptocurrency" appears 2 times in current statement
- Appears in 5 out of 100 statements

**Calculations:**
- TF-IDF(inflation) = $80 \times \log(100/95) = 80 \times 0.051 = 4.1$
- TF-IDF(cryptocurrency) = $2 \times \log(100/5) = 2 \times 3.00 = 6.0$

---

## Word Embeddings: Motivation

**BoW and TF-IDF limitations:**
- No semantic similarity
- "King" and "Queen" are as different as "King" and "Apple"

**Word embeddings:**
- Map each word to dense vector (e.g., 300 dimensions)
- Similar words have similar vectors
- Captures semantic relationships

---

## Word2Vec: Skip-Gram Model

**Training objective:**
- Given word $w$, predict context words
- "The **inflation** rate is rising" → predict {the, rate, is, rising}

**Neural network approach:**
- Input: one-hot encoded word
- Hidden layer: word embedding (e.g., 300-dim)
- Output: probabilities for context words

**Result:** Words in similar contexts get similar embeddings

---

## Word Embeddings: Semantic Arithmetic

<!-- Suggested image: Word embeddings 2D projection showing semantic relationships between economic terms -->

**Famous example:**
$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

**Economic examples:**
$$\vec{inflation} - \vec{high} + \vec{low} \approx \vec{deflation}$$

$$\vec{recession} - \vec{bad} + \vec{good} \approx \vec{expansion}$$

**Applications:**
- Find similar terms in corpus
- Measure concept distances

---

## GloVe: Global Vectors

**Alternative to Word2Vec:**
- Train on word co-occurrence matrix
- Factorization approach

**Key equation:**
$$\vec{w}_i^T \vec{w}_j = \log P(w_j | w_i)$$

**Pre-trained embeddings available:**
- Common Crawl (840B tokens)
- Wikipedia + Gigaword (6B tokens)

---

## Contextual Embeddings: BERT

**Problem with Word2Vec/GloVe:**
- Each word has one embedding
- "bank" (financial) vs. "bank" (river) → same vector!

**BERT (Bidirectional Encoder Representations from Transformers):**
- Embeddings depend on context
- "I went to the bank to deposit money" → financial meaning
- "I sat by the river bank" → geographical meaning

---

## BERT Architecture

**Transformer-based:**
- Self-attention mechanism
- Processes entire sentence bidirectionally
- Pre-trained on massive corpora (Wikipedia, Books)

**Two training tasks:**
1. Masked Language Model: predict masked words
2. Next Sentence Prediction: determine if two sentences are consecutive

**Output:** Contextual embedding for each token

---

## Using BERT for Economic Text

**Example: Sentiment of FOMC statements**

**Approach:**
1. Fine-tune BERT on labeled financial text
2. Input: FOMC paragraph
3. Output: hawkish/dovish classification + confidence

**Performance:**
- State-of-the-art accuracy (90%+)
- Beats dictionary methods significantly

---

## Sentiment Analysis

<!-- Suggested image: Sentiment analysis time series showing text sentiment scores over time -->

**Goal:** Extract emotional tone from text

**Applications in economics:**
- Consumer confidence from tweets
- Policy uncertainty from news
- Financial market sentiment

**Methods:**
1. Dictionary-based (Loughran-McDonald)
2. Supervised ML (train classifier on labeled data)

---

## Dictionary Methods: Loughran-McDonald

**Finance-specific word lists:**
- Positive: {growth, profit, gain, strong, ...}
- Negative: {loss, decline, weak, risk, ...}
- Uncertainty: {uncertain, doubt, risk, volatile, ...}

**Sentiment score:**
$$\text{Sentiment} = \frac{\#\text{positive} - \#\text{negative}}{\#\text{total words}}$$

**Widely used for 10-K filings, earnings calls**

---

## Example: Earnings Call Sentiment

**Firm A's earnings call excerpt:**
"We experienced strong revenue growth despite challenging market conditions. Profits exceeded expectations, though uncertainty remains about future demand."

**Counts:**
- Positive: {strong, growth, exceeded} = 3
- Negative: {challenging, uncertainty} = 2
- Total words: 20

**Sentiment:** $(3-2)/20 = 0.05$ (slightly positive)

---

## Supervised Sentiment Analysis

**Training data:**
- Manually label documents as positive/negative/neutral
- Features: TF-IDF, BERT embeddings

**Classification algorithms:**
- Logistic regression
- Random forests
- Fine-tuned BERT

**Advantage:** Learns domain-specific patterns beyond dictionary

---

## Topic Modeling: LDA

**Latent Dirichlet Allocation (Blei et al. 2003):**
- Automatically discover topics in corpus
- Each document is mixture of topics
- Each topic is distribution over words

**Example: FOMC minutes**
- Topic 1 (Inflation): {inflation, prices, CPI, 2%, target, ...}
- Topic 2 (Employment): {unemployment, jobs, labor, wages, ...}
- Topic 3 (Financial Stability): {banks, credit, financial, risk, ...}

---

## LDA: Generative Model

**For each document $d$:**
1. Draw topic proportions $\theta_d \sim \text{Dirichlet}(\alpha)$
2. For each word $n$ in document:
   - Draw topic $z_n \sim \text{Categorical}(\theta_d)$
   - Draw word $w_n \sim \text{Categorical}(\beta_{z_n})$

**Inference:** Given observed words, infer topics $\beta$ and proportions $\theta$

**Use variational inference or Gibbs sampling**

---

## LDA: Practical Example

**Corpus: 1,000 Fed speeches (2000-2023)**
- Choose $K = 5$ topics
- Run LDA algorithm

**Discovered topics:**
1. Monetary policy tools (interest, rates, QE)
2. Economic outlook (growth, GDP, forecast)
3. Financial stability (banks, regulation, crisis)
4. Labor markets (unemployment, jobs, wages)
5. Inflation (prices, CPI, target)

---

## Topic Evolution Over Time

**Application: Track topic prevalence**

For each year $t$:
$$\text{Topic}_k(t) = \frac{1}{N_t} \sum_{d \in t} \theta_{d,k}$$

**Finding:**
- Topic "Financial Stability" spikes in 2008-2009
- Topic "Inflation" rises in 2021-2023
- Provides quantitative measure of policy focus

---

## Causal Inference with Text: Text as Confounder

**Example: Social media and voting**
- Treatment $W$: exposure to political ad
- Outcome $Y$: voting behavior
- Confounder: user's past tweet content

**Challenge:**
- Must control for tweet content
- Text is high-dimensional

---

## DML with Text Features

**Approach:**
1. Embed tweets using BERT → vectors $T_i \in \mathbb{R}^{768}$
2. Include $T_i$ in covariate set $X_i$
3. Run DML:
   - $\hat{m}(X_i, T_i) = E[Y | X_i, T_i]$
   - $\hat{\ell}(X_i, T_i) = E[W | X_i, T_i]$
4. Estimate: $\hat{\theta} = \frac{\sum (Y_i - \hat{m})(W_i - \hat{\ell})}{\sum (W_i - \hat{\ell})^2}$

---

## Text as Outcome

**Research question:** How does policy change discourse?

**Example: Baker, Bloom, Davis (2016) - Economic Policy Uncertainty**
- Count news articles mentioning "uncertainty" and "economy"
- Index spikes during crises
- Used as outcome in event studies

**Causal question:**
- Does Fed communication affect media uncertainty coverage?
- Treatment: FOMC statement tone
- Outcome: EPU index

---

## Text as Treatment

**Example: Measuring policy stance**

**Approach:**
1. Quantify hawkish/dovish tone of FOMC statement
2. Treat as continuous treatment variable
3. Outcome: financial market reactions

**Dose-response function:**
$$\mu(t) = E[Y | \text{Text} = t]$$

where $t$ represents degree of hawkishness

---

## Named Entity Recognition (NER)

**Goal:** Extract specific entities from text
- People: "Janet Yellen"
- Organizations: "Federal Reserve"
- Locations: "United States"
- Dates: "March 2023"

**Applications:**
- Track mentions of firms in news
- Identify central bank officials
- Geographic coverage in media

---

## NER: Economic Application

**Firm mentions in news and stock returns:**

**Research design:**
1. Run NER on financial news corpus
2. Count mentions of each firm per day
3. Measure sentiment of sentences containing firm name
4. Regress stock returns on mention count + sentiment

**Finding:** Negative news mentions predict negative returns

---

## Large Language Models (LLMs)

**GPT-3, GPT-4, Claude, etc.:**
- Billions of parameters
- Trained on massive text corpora
- Can perform zero-shot tasks

**Economic applications:**
- Summarize earnings calls
- Classify central bank stance
- Extract economic indicators from text

---

## Example: LLM for Economic Measurement

**Prompt to GPT-4:**
"Read this FOMC statement and classify the monetary policy stance as: (1) Very Dovish, (2) Dovish, (3) Neutral, (4) Hawkish, (5) Very Hawkish. Explain your reasoning."

**Advantage:**
- No labeled training data needed
- Can handle complex, nuanced text
- Provides interpretable reasoning

---

## Challenges with Text Data

**Preprocessing:**
- Tokenization (splitting into words)
- Stemming/Lemmatization (run → running → ran)
- Stop word removal (the, is, at)
- Language-specific issues (Chinese, Arabic)

**Statistical challenges:**
- High dimensionality
- Sparsity
- Measurement error in embeddings

---

## Chinese Text Processing: Jieba

**Challenge: No spaces between words**
- English: "economic growth"
- Chinese: "经济增长" (needs segmentation)

**Jieba (结巴) library:**
```python
import jieba
text = "经济增长速度加快"
words = jieba.cut(text)
# Output: ['经济', '增长', '速度', '加快']
```

---

## Project Idea: Digital Transformation

**Research question:**
- Does firm's digital transformation affect productivity?

**Data:**
- Chinese listed firms' annual reports (available on CNINFO)
- Keywords: "数字化", "大数据", "人工智能", "云计算"

**Method:**
1. Count digital keywords (TF-IDF weighted)
2. Measure TFP from financial data
3. DML with text controls for firm characteristics

---

## Text Data Sources

**Public sources:**
- EDGAR (SEC filings)
- Federal Reserve speeches and minutes
- News archives (LexisNexis, Factiva)
- Social media (Twitter API, Reddit)

**Chinese sources:**
- CNINFO (巨潮资讯网): annual reports
- Eastmoney (东方财富): financial news
- Weibo API: social media

---

## Software Ecosystem: NLP

**Python libraries:**
- `nltk`: Natural Language Toolkit (basic processing)
- `spacy`: Industrial-strength NLP
- `gensim`: Topic modeling (LDA, Word2Vec)
- `transformers`: HuggingFace (BERT, GPT)

**R packages:**
- `quanteda`: Quantitative analysis of textual data
- `text2vec`: Word embeddings
- `tidytext`: Tidy text mining

---

## Combining Causal Inference and NLP

**The frontier:**
1. Text controls for confounding
2. Text as outcome or treatment
3. Heterogeneous effects by textual features

**Example research:**
- Gentzkow & Shapiro (2010): Media slant and persuasion
- Hassan et al. (2019): Firm-level political risk
- Bybee et al. (2020): Structural interpretation of text-based risks

---

## Key Readings: Text as Data

**Gentzkow, Kelly, Taddy (2019):**
- "Text as Data" in *Journal of Economic Literature*
- Comprehensive survey
- Essential reference

**Grimmer & Stewart (2013):**
- "Text as Data: The Promise and Pitfalls"
- Political Science Applications Review
- Methodological guidance

---

## Key Takeaways: Text as Data

**Main insights:**
- Text contains rich economic information
- Multiple representation methods (BoW → BERT)
- Can be outcome, treatment, or confounder
- Combines with causal inference methods

**Challenges:**
- High dimensionality
- Preprocessing complexity
- Validation and interpretation

---

<!-- _class: lead -->

# Part 3
## Course Synthesis

---

## The ML for Economics Pipeline

**Complete workflow:**
1. **Prediction:** Machine learning for $\hat{y}$
2. **Identification:** DAGs, CIA, IV for causal structure
3. **Estimation:** DML, Causal Forests for $\hat{\tau}(x)$
4. **Decision:** Policy learning for optimal allocation
5. **New data types:** Text, images, networks

**This course covered the full pipeline!**

---

## From Prediction to Causation to Decision

**Prediction (Lectures 1-4):**
- Regularization, cross-validation, neural networks
- Goal: forecast $Y$ from $X$

**Causation (Lectures 5-9):**
- Selection on observables, DAGs, IV, DML
- Goal: estimate $E[Y(1) - Y(0) | X]$

**Decision (Lecture 10):**
- Policy learning, welfare maximization
- Goal: choose optimal $\pi(X)$

---

## When to Use Each Method

| Problem | Method | Example |
|---------|--------|---------|
| Prediction | Lasso, RF, NN | Credit scoring |
| ATE estimation | DML, DR | Program evaluation |
| HTE estimation | Causal Forest | Personalized medicine |
| Policy choice | Policy Trees | Targeting interventions |
| Unstructured data | NLP, CV | Sentiment analysis |

---

## Final Project Suggestions

**High-impact topics:**
1. Digital transformation and firm performance (with text)
2. Environmental policy and corporate disclosure
3. Social media sentiment and consumer behavior
4. Heterogeneous treatment effects in education/health
5. Optimal targeting in development programs

**Requirements:** Clear causal question + appropriate method

---

## Data Resources for Projects

**Chinese administrative data:**
- CSMAR: listed firm financials
- CNRDS: research-ready datasets
- Provincial statistics

**International:**
- World Bank microdata
- IPUMS (census/survey data)
- Academic Data Services

**Use your comparative advantage!**

---

## Methodological Best Practices

**For credible empirical work:**
1. Pre-register analysis plan
2. Report all specifications (not just significant ones)
3. Validate on hold-out data
4. Conduct sensitivity analyses
5. Make code/data available

**Transparency builds trust**

---

## Looking Forward: Frontier Topics

**Beyond this course:**
- Reinforcement learning for dynamic treatment regimes
- Synthetic controls and panel methods
- Network causal inference
- Causal discovery (learning DAGs from data)
- Computer vision for economic measurement

**The field is rapidly evolving!**

---

## Resources for Continued Learning

**Books:**
- Cunningham (2021): *Causal Inference: The Mixtape*
- Huntington-Klein (2022): *The Effect*
- Angrist & Pischke (2009): *Mostly Harmless Econometrics*

**Online:**
- NBER Summer Institute lectures
- Mixtape Sessions (workshops)
- YouTube: Ben Lambert, Brady Neal

---

## Final Thoughts

**Machine learning is transforming economics:**
- New data sources (text, images, networks)
- Better prediction and heterogeneity estimation
- Bridging to decision-making

**But fundamentals remain:**
- Clear causal questions
- Identification assumptions
- Careful validation

**Economics + ML = powerful combination**

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Final Projects**: Due [Date]

See you next time!
