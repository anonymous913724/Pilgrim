# PILGRIM

In the following sections, we detail some basic instructions on how to setup the environment for PILGRIM and run the core experiment files.

## Setup

1. Run the setup script to create a venv and install dependencies.

```bash
./setup.sh
```

2. Activate the environment in your current shell.

```bash
source venv/bin/activate
```

## Run

Run one of the provided scripts. These call the Python entry points with the expected arguments.

```bash
./run_compare_covid.sh
```

```bash
./run_compare_covid_unify.sh
```

```bash
./run_compare_pems_faiss.sh
```

```bash
./run_compare_yelp.sh
```

## Command Line Arguments

If you wish, you can also run these experiment files manually. Below is a list of experiment scripts you can run and their command line arguments (if applicable). You should run these from the root directory of this project, e.g.:

```bash
python3 soft_topk_attn/compare_pems_faiss.py
```

### compare_covid_baseline.py

DGNN-only baseline for COVID county prediction.

```
--nyt_csv TEXT              NYT us-counties.csv (required)
--adj_txt TEXT              Census county adjacency txt (required)
--date_start TEXT           Start date (default: 2021-04-30)
--date_end TEXT             End date (default: 2022-04-30)
--snapshot_days INT         Days per snapshot (default: 7)
--feature_mode TEXT         cases_only or cases_deaths (default: cases_only)
--train_end_start TEXT      Train end date (default: 2022-01-30)
--model TEXT                DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                       Use CPU instead of GPU
--lags INT                  Temporal lags (default: 1)
--horizon INT               Prediction horizon (default: 1)
--k_hop INT                 K-hop subgraph size (default: 2)
--q_thr FLOAT               Quantile threshold for labels (default: 0.9)
--d_emb INT                 Embedding dimension (default: 64)
--dcrnn_k INT               DCRNN K parameter (default: 2)
--lr FLOAT                  Learning rate (default: 1e-3)
--anchors_train INT         Training anchors per snapshot (default: 512)
--anchors_eval INT          Eval anchors per snapshot (default: 512)
--k_eval INT                K for evaluation metrics (default: 50)
--roll_history INT          Rolling window history (-1=all, default: -1)
--roll_epochs INT           Epochs per rolling window (default: 1)
--burnin_eval INT           Skip first B snapshots (default: 0)
--seed INT                  Random seed (default: 0)
```

### compare_covid_baseline_unify.py

DGNN-only baseline for COVID with unified training.

```
--nyt_csv TEXT              NYT us-counties.csv (required)
--adj_txt TEXT              Census county adjacency txt (required)
--date_start TEXT           Start date (default: 2021-04-30)
--date_end TEXT             End date (default: 2022-04-30)
--snapshot_days INT         Days per snapshot (default: 7)
--feature_mode TEXT         cases_only or cases_deaths (default: cases_only)
--train_end_start TEXT      Train end date (default: 2022-04-30)
--model TEXT                DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                       Use CPU instead of GPU
--lags INT                  Temporal lags (default: 1)
--horizon INT               Prediction horizon (default: 1)
--k_hop INT                 K-hop subgraph size (default: 2)
--q_thr FLOAT               Quantile threshold for labels (default: 0.9)
--tau_roll_window INT       Rolling window for threshold (default: 8)
--tau_thr_init FLOAT        Initial threshold (default: 0.0)
--d_emb INT                 Embedding dimension (default: 64)
--dcrnn_k INT               DCRNN K parameter (default: 2)
--lr FLOAT                  Learning rate (default: 1e-3)
--anchors_train INT         Training anchors per snapshot (default: 512)
--anchors_eval INT          Eval anchors per snapshot (default: 512)
--k_eval INT                K for evaluation metrics (default: 50)
--roll_history INT          Rolling window history (-1=all, default: -1)
--roll_epochs INT           Epochs per rolling window (default: 1)
--burnin_eval INT           Skip first B snapshots (default: 0)
--seed INT                  Random seed (default: 0)
```

### compare_covid.py

Full model with attention and optional FAISS for COVID.

```
--nyt_csv TEXT              NYT us-counties.csv (required)
--adj_txt TEXT              Census county adjacency txt (required)
--date_start TEXT           Start date (default: 2021-04-30)
--date_end TEXT             End date (default: 2022-04-30)
--snapshot_days INT         Days per snapshot (default: 7)
--feature_mode TEXT         cases_only or cases_deaths (default: cases_only)
--train_end_start TEXT      Train end date (default: 2022-01-30)
--holdout_snaps INT         Internal holdout size (default: 0)
--model TEXT                DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                       Use CPU instead of GPU
--lags INT                  Temporal lags (default: 1)
--horizon INT               Prediction horizon (default: 1)
--k_hop INT                 K-hop subgraph size (default: 2)
--q_thr FLOAT               Quantile threshold for labels (default: 0.9)
--d_emb INT                 Embedding dimension (default: 64)
--dcrnn_k INT               DCRNN K parameter (default: 2)
--lr FLOAT                  Learning rate (default: 1e-3)
--anchors_train INT         Training anchors per snapshot (default: 512)
--anchors_eval INT          Eval anchors per snapshot (default: 512)
--k_eval INT                K for evaluation metrics (default: 50)
--warmup_snaps INT          Warmup snapshots (default: 4)
--epochs_warmup INT         Warmup epochs (default: 1)
--anneal FLOAT              Annealing rate (default: 0.9)
--tau FLOAT                 Temperature parameter (default: 0.8)
--init_k_frac FLOAT         Initial k fraction (default: 0.05)
--k_min FLOAT               Min k (default: 0.01)
--k_max FLOAT               Max k (default: 0.2)
--k_abs_min INT             Min absolute k (default: 20)
--k_abs_max INT             Max absolute k (default: 50)
--beta_div FLOAT            Diversity loss weight (default: 0.1)
--beta_metric FLOAT         Metric loss weight (default: 0.5)
--beta_mixup FLOAT          Mixup weight (default: 1.0)
--use_faiss                 Enable FAISS candidate retrieval
--faiss_metric TEXT         FAISS metric: ip or l2 (default: ip)
--faiss_update_every INT    Rebuild FAISS every N snapshots (default: 1)
--faiss_c FLOAT             Candidate multiplier (default: 30.0)
--faiss_topm_init INT       Warmup init topm (default: 0)
--faiss_topm_min INT        Minimum topm (default: 64)
--faiss_topm_max INT        Maximum topm (default: 4096)
--faiss_max_cand INT        Cap candidate set size (default: 4096)
--faiss_union_khop          Union FAISS candidates with k-hop
--faiss_require_torch_gpu   Hard-fail if torch GPU unavailable
--roll_history INT          Rolling window history (-1=all, default: -1)
--roll_epochs INT           Epochs per rolling window (default: 1)
--burnin_eval INT           Skip first B snapshots (default: 0)
```

### compare_covid_unify.py

Full model with attention and optional FAISS with unified training.

```
--nyt_csv TEXT              NYT us-counties.csv (required)
--adj_txt TEXT              Census county adjacency txt (required)
--date_start TEXT           Start date (default: 2021-04-30)
--date_end TEXT             End date (default: 2022-04-30)
--snapshot_days INT         Days per snapshot (default: 7)
--feature_mode TEXT         cases_only or cases_deaths (default: cases_only)
--train_end_start TEXT      Train end date (default: 2022-04-30)
--model TEXT                DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                       Use CPU instead of GPU
--lags INT                  Temporal lags (default: 1)
--horizon INT               Prediction horizon (default: 1)
--k_hop INT                 K-hop subgraph size (default: 2)
--q_thr FLOAT               Quantile threshold for labels (default: 0.9)
--d_emb INT                 Embedding dimension (default: 64)
--dcrnn_k INT               DCRNN K parameter (default: 2)
--lr FLOAT                  Learning rate (default: 1e-3)
--anchors_train INT         Training anchors per snapshot (default: 512)
--anchors_eval INT          Eval anchors per snapshot (default: 512)
--k_eval INT                K for evaluation metrics (default: 50)
--warmup_snaps INT          Warmup snapshots (default: 4)
--epochs_warmup INT         Warmup epochs (default: 1)
--anneal FLOAT              Annealing rate (default: 0.9)
--tau FLOAT                 Temperature parameter (default: 0.8)
--init_k_frac FLOAT         Initial k fraction (default: 0.05)
--k_min FLOAT               Min k (default: 0.01)
--k_max FLOAT               Max k (default: 0.2)
--k_abs_min INT             Min absolute k (default: 20)
--k_abs_max INT             Max absolute k (default: 50)
--beta_div FLOAT            Diversity loss weight (default: 0.1)
--beta_metric FLOAT         Metric loss weight (default: 0.5)
--beta_mixup FLOAT          Mixup weight (default: 1.0)
--use_faiss                 Enable FAISS candidate retrieval
--faiss_metric TEXT         FAISS metric: ip or l2 (default: ip)
--faiss_update_every INT    Rebuild FAISS every N snapshots (default: 1)
--faiss_c FLOAT             Candidate multiplier (default: 30.0)
--faiss_topm_init INT       Warmup init topm (default: 0)
--faiss_topm_min INT        Minimum topm (default: 64)
--faiss_topm_max INT        Maximum topm (default: 4096)
--faiss_max_cand INT        Cap candidate set size (default: 4096)
--faiss_union_khop          Union FAISS candidates with k-hop
--faiss_require_torch_gpu   Hard-fail if torch GPU unavailable
--roll_history INT          Rolling window history (-1=all, default: -1)
--roll_epochs INT           Epochs per rolling window (default: 1)
--burnin_eval INT           Skip first B snapshots (default: 0)
```

### compare_pems_faiss.py

Hardcoded configuration for PEMS traffic data (no command line arguments). Edit constants at the top of the file to configure:

```
MODEL_NAME              DCRNN, SEHTGNN, or TASER
USE_ATTN               Enable attention model
USE_MIXUP              Enable mixup and metric loss
HORIZON                Prediction horizon (1 or 3)
LAGS, D_EMB, DCRNN_K   Model dimensions
K_HOP, THRESHOLD       Subgraph parameters
EPOCHS, LR             Training parameters
Various loss weights and FAISS settings
```

### yelp_baseline.py

DGNN-only baseline for Yelp reviewer prediction.

```
--pt TEXT               Preprocessed Yelp .pt (from yelp_process.py) (required)
--model TEXT            DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                   Use CPU instead of GPU
--lags INT              Temporal lags (default: 1)
--horizon INT           Prediction horizon (default: 1)
--test_horizons INT...  Test on multiple horizons (default: [1])
--recent_window INT     Months to define reviewers/peers (default: 6)
--quantile FLOAT        Threshold percentile (default: 0.5)
--min_reviewers INT     Min reviewers per business (default: 5)
--min_peers INT         Min peers per business (default: 5)
--k_hop INT             K-hop subgraph size (default: 2)
--d_emb INT             Embedding dimension (default: 32)
--dcrnn_k INT           DCRNN K parameter (default: 2)
--lr FLOAT              Learning rate (default: 1e-3)
--anchors_train INT     Training anchors per snapshot (default: 256)
--anchors_eval INT      Eval anchors per snapshot (default: 512)
--k_eval INT            K for evaluation metrics (default: 50)
--warmup_snaps INT      Warmup snapshots (default: 6)
--epochs_warmup INT     Warmup epochs (default: 3)
--print_every INT       Print progress every N steps (default: 1)
```

### yelp_full.py

Full attention model for Yelp with optional FAISS.

```
--pt TEXT               Preprocessed Yelp .pt (from yelp_process.py) (required)
--model TEXT            DCRNN, SEHTGNN, or TASER (default: SEHTGNN)
--cpu                   Use CPU instead of GPU
--lags INT              Temporal lags (default: 1)
--horizon INT           Prediction horizon (default: 1)
--test_horizons INT...  Test on multiple horizons (default: [1])
--recent_window INT     Months to define reviewers/peers (default: 6)
--quantile FLOAT        Threshold percentile (default: 0.5)
--min_reviewers INT     Min reviewers per business (default: 5)
--min_peers INT         Min peers per business (default: 5)
--k_hop INT             K-hop subgraph size (default: 2)
--d_emb INT             Embedding dimension (default: 32)
--dcrnn_k INT           DCRNN K parameter (default: 2)
--lr FLOAT              Learning rate (default: 1e-3)
--anchors_train INT     Training anchors per snapshot (default: 256)
--anchors_eval INT      Eval anchors per snapshot (default: 512)
--k_eval INT            K for evaluation metrics (default: 50)
--warmup_snaps INT      Warmup snapshots (default: 6)
--epochs_warmup INT     Warmup epochs (default: 3)
--anneal FLOAT          Annealing rate (default: 0.9)
--print_every INT       Print progress every N steps (default: 1)
--tau FLOAT             Temperature parameter (default: 0.6)
--init_k_frac FLOAT     Initial k fraction (default: 0.05)
--k_min FLOAT           Min k (default: 0.01)
--k_max FLOAT           Max k (default: 0.2)
--k_abs_min INT         Min absolute k (default: 10)
--k_abs_max INT         Max absolute k (default: 50)
--beta_div FLOAT        Diversity loss weight (default: 0.0)
--beta_metric FLOAT     Metric loss weight (default: 0.5)
--beta_mixup FLOAT      Mixup weight (default: 1.0)
```

### yelp_full_faiss.py

Full attention model with FAISS candidate retrieval for Yelp.

```
--pt TEXT               Preprocessed Yelp .pt (from yelp_process.py) (required)
--model TEXT            DCRNN, SEHTGNN, or TASER (default: TASER)
--cpu                   Use CPU instead of GPU
--lags INT              Temporal lags (default: 1)
--horizon INT           Prediction horizon (default: 1)
--test_horizons INT...  Test on multiple horizons (default: [1])
--recent_window INT     Months to define reviewers/peers (default: 6)
--quantile FLOAT        Threshold percentile (default: 0.5)
--min_reviewers INT     Min reviewers per business (default: 5)
--min_peers INT         Min peers per business (default: 5)
--k_hop INT             K-hop subgraph size (default: 2)
--d_emb INT             Embedding dimension (default: 32)
--dcrnn_k INT           DCRNN K parameter (default: 2)
--lr FLOAT              Learning rate (default: 1e-3)
--anchors_train INT     Training anchors per snapshot (default: 256)
--anchors_eval INT      Eval anchors per snapshot (default: 512)
--k_eval INT            K for evaluation metrics (default: 50)
--warmup_snaps INT      Warmup snapshots (default: 6)
--epochs_warmup INT     Warmup epochs (default: 3)
--anneal FLOAT          Annealing rate (default: 0.95)
--print_every INT       Print progress every N steps (default: 1)
--tau FLOAT             Temperature parameter (default: 0.6)
--init_k_frac FLOAT     Initial k fraction (default: 0.05)
--k_min FLOAT           Min k (default: 0.01)
--k_max FLOAT           Max k (default: 0.2)
--k_abs_min INT         Min absolute k (default: 10)
--k_abs_max INT         Max absolute k (default: 50)
--beta_div FLOAT        Diversity loss weight (default: 0.1)
--beta_metric FLOAT     Metric loss weight (default: 0.5)
--beta_mixup FLOAT      Mixup weight (default: 1.0)
--use_faiss             Enable FAISS candidate retrieval
--faiss_metric TEXT     FAISS metric: ip or l2 (default: ip)
--faiss_update_every INT Rebuild FAISS every N snapshots (default: 1)
--faiss_c FLOAT         Candidate multiplier (default: 10.0)
--faiss_topm_init INT   Warmup init topm (default: 0)
--faiss_topm_min INT    Minimum topm (default: 64)
--faiss_topm_max INT    Maximum topm (default: 4096)
--faiss_max_cand INT    Cap candidate set size (default: 2048)
--faiss_union_khop      Union FAISS candidates with k-hop
--faiss_require_torch_gpu Hard-fail if torch GPU unavailable
```

## Hardcoded Configuration Files

The following files use hardcoded configuration constants instead of command line arguments. Edit the constants at the top of each file to configure:

### compare_yelp_simple.py
Yelp business prediction with simplified attention model.

### compare_yelp.py
Yelp business prediction with full attention and FAISS.

### lowm_pems_baseline.py
DCRNN baseline for PEMS traffic (node-level).

### lowm_pems_attn.py
Attention model for PEMS traffic (node-level).

### multi_pems_compare.py
Multi-snapshot PEMS traffic comparison.

### multi_pems_compare_unify.py
Multi-snapshot PEMS traffic with unified training.

### multi_pems_attn.py
Multi-snapshot PEMS with attention model.

### multi_pems_attn_unify.py
Multi-snapshot PEMS with attention and unified training.

