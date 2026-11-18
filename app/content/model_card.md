# Model Card: Telecom Churn Decision Support

### 1. Model Overview & Architecture
This model identifies telecom customers at elevated risk of churn to support cost-effective retention campaigns. The core estimator is a **tuned LightGBM classifier** configured with balanced class weighting, shallow tree depth (`max_depth=3`, `num_leaves=7`), and a modest learning rate to prevent overfitting on tabular features. Outputs are calibrated using **Sigmoid (Platt) calibration**, aligning predicted probabilities with true empirical churn frequencies.

The architecture was selected by evaluating eight candidate pipelines via 5-fold stratified cross-validation on training data, optimizing primarily for Area Under the Precision-Recall Curve (**AUC-PR**).

### 2. Training & Evaluation Data
- **Source**: Public IBM Telco Customer Churn dataset comprising 7,043 unique residential customer accounts.
- **Structure**: Cross-sectional snapshot with demographic attributes, subscribed services, account tenure, contract terms, payment methods, and monthly charges. Contains no event dates or temporal logging.
- **Split Protocol**: An 80/20 stratified split separated 5,634 training rows from 1,409 held-out test rows. The test set was locked prior to modeling and evaluated exactly once for final reporting. Training rows use out-of-fold cross-validation estimates.

### 3. Decision Cutoff & Economics
The probability cutoff is derived mathematically from retention campaign economics rather than heuristic threshold tuning:
$$p^* = \frac{d}{d + r(1 - d)}$$
Under default assumptions—a retention success rate $r = 50\%$ and a discount rate $d = 20\%$ across a 12-month horizon—the optimal cutoff is strictly **$1/3 \approx 0.333$**. Customers with calibrated churn probability $p \ge 1/3$ produce positive expected net value when contacted.

### 4. Headline Performance (Held-out Test Set, N=1,409)
- **Discrimination**: AUC-PR of **0.655** (against a 26.5% baseline churn rate) and ROC-AUC of **0.842**.
- **Probability Calibration**: Brier score improves from **0.166** (uncalibrated) to **0.137** after Platt scaling, ensuring probabilities represent reliable percentage risks.
- **Operational Metrics**: At $p^* = 0.333$, the model flags **496** customers (35.2% contact rate), capturing **72.5%** of all churners (271 of 374) with **54.6%** precision.
- **Net Economic Value**: Yields a realized net gain of **$60,674**, outperforming an untargeted discount policy by +$17,595 (+40.8%, 95% CI [$8,676, $27,360]).

### 5. Limitations & Caveats
1. **Churn Prediction vs. Uplift**: The model predicts the probability of churn, not the customer's responsiveness to an offer.
2. **Fixed Success Assumption**: Assumes a constant retention probability ($r=0.50$); actual campaign response varies by segment.
3. **No Temporal Dimension**: Historical snapshot lacks seasonality, customer lifecycle events, or macroeconomic signals.
4. **Sample Size Uncertainty**: With 374 churners in the held-out test split, sampling uncertainty remains moderate.
5. **Optimistic Out-of-Fold Scores**: Full-database training scores reflect out-of-fold estimates, which are slightly optimistic relative to unseen future data.

### 6. Monitoring & Intended Use
Input data health is enforced at inference time via strict schema validation, range checks, and category verification. Intended strictly as a decision-support and portfolio demonstration system, not for automated contract alterations or real customer penalties.
