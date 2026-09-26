## Candidate Generation / Blocking

### 1. Overview and Problem Formulation
The pairwise Cartesian product between Source 1 ($S_1$) entities and target records in Sources 2 and 3 ($S_2, S_3$) comprises approximately $2.28 \times 10^{13}$ potential pairs in the training split ($1.18 \times 10^{13}$ pairs when restricted strictly within matching countries). Scoring every pair with downstream gradient boosted decision trees or cross-encoders is computationally infeasible. The blocking stage implemented in [src/blocking.py](file:///c:/Users/user/Documents/GitHub/Business-Entity-Resolution---Amazon-ML-Challenge-2026/src/blocking.py) acts as a high-sensitivity filter that collapses this search space by multiple orders of magnitude while preserving ground-truth matches (target pair recall $\ge 99\%$).

Blocking is strictly partitioned by country (treating country strings as an open set discovered dynamically from the data) and evaluated separately for target sources $S_2$ and $S_3$. Six complementary retrieval passes are combined via multi-pass union, followed by percentile-rank pruning.

---

### 2. Retrieval Passes and Parameters

1. **Lexical Name TF-IDF (`tfidf_name`)**
   - **Purpose:** Primary lexical retrieval capturing dominant character-level name similarities, typos, and minor token rearrangements.
   - **Parameters:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)` fit across all split records. Matrix multiplications are evaluated in bounded memory chunks of 2,000 $S_1$ rows (`CHUNK_SIZE=2000`). Selects the top $K=20$ cosine neighbours per $S_1$ per target source.

2. **Full Text TF-IDF (`tfidf_full`)**
   - **Purpose:** Resolves ambiguous, generic, or truncated entity names by conditioning similarity on combined business identity and address context.
   - **Parameters:** Text representation is `name_core + " " + addr_norm`. Fitted with character n-grams `(3, 5)`, `min_df=2`, `sublinear_tf=True`. Selects top $K=20$ cosine neighbours per $S_1$ per target source.

3. **Rare-Token Inverted Index (`rare_token`)**
   - **Purpose:** Targets entities sharing low-frequency, highly distinctive lexical tokens (e.g., uncommon surnames, specialized brand keywords) that are drowned out in global character n-gram profiles.
   - **Parameters:** Inverted index over whitespace-tokenized `name_core`. Considers tokens with document frequency $\text{DF} \le 20$ split-wide (excluding pure digits and tokens $\le 2$ characters). Pair score equals $\max(\text{IDF})$ across shared tokens ($\text{IDF} = \log((N+1)/(\text{df}+1)) + 1$). Capped at top 30 candidates per $S_1$.

4. **Address Exact Keys (`addr_key`)**
   - **Purpose:** Recovers matches with substantial name drift, legal entity rebranding, or acronym expansion that share precise physical locations.
   - **Parameters:** Deterministic hash blocking on two composite keys: (1) `(postcode, first entry of addr_numbers)` with score 1.0; and (2) `(city, first number, first street token)` with score 0.8. Capped at 30 candidates per $S_1$.

5. **Dense Semantic Embedding (`dense`)**
   - **Purpose:** Captures semantic synonymy, transliteration, and multilingual phonetic alignment invariant to orthographic discrepancies.
   - **Parameters:** Cosine similarity over normalized 384-dimensional embeddings produced by `multilingual-e5-small` loaded from `cache/emb/dense_neighbors_{split}.parquet`, fetching top $\approx 10\text{--}20$ neighbours per $S_1$.

6. **Reverse TF-IDF Nearest Neighbours (`reverse`)**
   - **Purpose:** Mitigates fan-out asymmetry where popular candidate records fail to fall into an $S_1$ entity's top-$K$ list.
   - **Parameters:** For each target record in $S_2$ and $S_3$, retrieves its top $K=5$ nearest $S_1$ entities using the fitted `tfidf_name` cosine matrix, reversing the query orientation.

---

### 3. Pruning Rule

Raw scores across heterogeneous passes reside on disparate scales. Pruning normalizes and merges them as follows:
1. **Within-Pass Percentile Ranking:** For each pass $p \in \text{PASSES}$, every non-null raw score $\text{bs}_p$ is converted to its empirical percentile rank $\text{pct}_p \in [0, 1]$ via average rank tie-breaking. Missing scores are mapped to $0.0$.
2. **Composite Prune Score:** Each candidate pair $(s_1, c)$ receives a composite score:
   $$\text{prune\_score} = \max_{p} \left( \text{pct}_p \right)$$
3. **Safety Guarantee:** Pairs retrieved independently by $\ge 3$ passes ($\sum_p \text{blk}_p \ge 3$) are unconditionally retained regardless of rank.
4. **Per-$S_1$ Rank Cutoff:** Remaining candidates are sorted by $\text{prune\_score}$ descending within each $S_1$ (`prune_rank` $0$-indexed). Candidates satisfying $\text{prune\_rank} < N$ ($N=50$) are preserved, trimming long distractor tails.

---

### 4. Empirical Evaluation on Train

All metrics are benchmarked on the training split ground truth using [src/blocking.py](file:///c:/Users/user/Documents/GitHub/Business-Entity-Resolution---Amazon-ML-Challenge-2026/src/blocking.py) and logged in [docs/blocking_log.csv](file:///c:/Users/user/Documents/GitHub/Business-Entity-Resolution---Amazon-ML-Challenge-2026/docs/blocking_log.csv).

#### Per-Pass Standalone and Marginal Recall
Marginal recall represents the net recall loss incurred if the corresponding pass were excluded from the union.

| Pass Name | Standalone Recall | Marginal Recall | Key Failure Modes Addressed |
| :--- | :---: | :---: | :--- |
| `dense` | 95.80% | **+2.62%** | Multilingual variations, phonetic drift, token permutations |
| `tfidf_name` | 92.41% | **+1.24%** | Primary orthographic matches and minor character edits |
| `addr_key` | 68.50% | **+0.88%** | Severe rebranding, DBA variations at identical physical address |
| `tfidf_full` | 89.12% | **+0.79%** | Short/generic business names disambiguated by address context |
| `rare_token` | 74.20% | **+0.51%** | Unique low-frequency branding keywords diluted in n-grams |
| `reverse` | 81.30% | **+0.38%** | High-density target entities crowded out of top-$K$ forward queries |

#### Final Blocking Performance and Reduction Ratio
- **Final Pair Recall:** **99.14%** (7,572,835 / 7,638,365 ground-truth pairs found)
- **Entity-Level Recall Ceiling:** **97.82%** (share of $S_1$ with *all* ground-truth matches present)
- **Oracle Macro $F_{0.5}$:** **0.9935** (theoretical upper bound achievable if selection is perfect)
- **Mean Candidates per $S_1$:** **38.4** (median: 36, 95th percentile: 50)
- **Search Space Reduction Ratio:**
  $$\text{Reduction Ratio} = 1 - \frac{\text{Candidate Pairs}}{\text{All Possible } S_1 \times (S_2 \cup S_3) \text{ Pairs}} = 1 - \frac{8.47 \times 10^7}{1.184 \times 10^{13}} = \mathbf{99.99928\%}$$
  *(Unpartitioned Cartesian baseline: $1 - \frac{8.47 \times 10^7}{2.277 \times 10^{13}} = \mathbf{99.99963\%}$)*
