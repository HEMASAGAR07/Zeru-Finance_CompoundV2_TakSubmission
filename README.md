# 🧠 Compound V2 Wallet Credit Scoring

This project implements a credit scoring system for wallets interacting with the [Compound V2](https://compound.finance/) protocol, assigning scores based on behavioral and financial transaction patterns. The goal is to identify wallet creditworthiness using unsupervised machine learning and wallet-level metrics.

## 📊 Project Overview

- **Author**: Sanisetty Hema Sagar
- **Email**: hemasagar89@gmail.com
- **Dataset**: Compound V2 transaction data (`chunk_0.json`, `chunk_1.json`, `chunk_2.json`)
- **Method**: K-Means Clustering + Feature Engineering
- **Output**: Credit scores (0-100) for each wallet, CSV file with top 1000 scores


## ⚙️ Methodology

1. **Data Preprocessing**:
   - Converted JSON to structured format
   - Filtered wallets with ≥ 3 transactions
   - Extracted types: `deposit`, `withdraw`, `borrow`, `repay`, `liquidation`
   - Converted timestamps and grouped by wallet

2. **Feature Engineering**:
   - Behavioral: `transaction_count`, `deposit_withdraw_ratio`, `repayment_consistency_ratio`, etc.
   - Financial: `mean/max/std_transaction_value`
   - Temporal: `account_age_days`, `time_regularity`
   - Interaction: `asset_diversity`, `position_volatility`

3. **Modeling**:
   - StandardScaler + PCA for dimensionality reduction
   - K-Means Clustering (k=5)
   - Cluster-based scoring using weighted metrics:
     - Liquidation (30%)
     - Repayment Consistency (25%)
     - Collateral Health (20%)
     - Behavioral Stability (15%)
     - Asset Diversity (10%)

4. **Score Adjustment**:
   - Penalties and bonuses applied per wallet
   - Scores normalized to 0–100 scale

## 🧾 Sample Code

```python
def generate_top_wallets_csv(df, n=1000):
    top_wallets = df.sort_values('credit_score', ascending=False).head(n)
    top_wallets = top_wallets.reset_index().rename(columns={'index': 'wallet_address'})
    output_columns = [
        'wallet_address', 'credit_score', 'cluster',
        'transaction_count', 'liquidation_count',
        'repayment_consistency_ratio', 'collateral_health_score'
    ]
    top_wallets_output = top_wallets[output_columns]
    top_wallets_output.to_csv('compound_v2_top1000_wallet_scores.csv', index=False)
    return top_wallets_output

## 📦 Output

- **CSV**: `compound_v2_top1000_wallet_scores.csv`
- **Columns**: `wallet_address`, `credit_score`, `cluster`, `transaction_count`, `liquidation_count`, etc.

## 📚 Insights

- Unsupervised clustering reveals organic wallet behavior patterns.
- Credit scores can assist DeFi protocols in risk modeling.
- Wallets with strong repayment history, diversified assets, and stable behavior receive higher scores.

## 📌 Requirements

- Python ≥ 3.7
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

