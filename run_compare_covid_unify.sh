#!/usr/bin/env bash
# Run exactly 6 tests: 3 baselines + 3 full models (no extras)

python3 soft_topk_attn/covid/compare_covid_baseline.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model DCRNN

python3 soft_topk_attn/covid/compare_covid_baseline.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model SEHTGNN

python3 soft_topk_attn/covid/compare_covid_baseline.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model TASER

python3 soft_topk_attn/covid/compare_covid.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --use_faiss \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model DCRNN

python3 soft_topk_attn/covid/compare_covid.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --use_faiss \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model SEHTGNN

python3 soft_topk_attn/covid/compare_covid.py \
  --nyt_csv data/nyt_covid/raw/us-counties.csv \
  --adj_txt data/nyt_covid/raw/county_adjacency.txt \
  --date_start 2021-11-30 \
  --date_end 2022-04-30 \
  --train_end_start 2022-04-30 \
  --burnin_eval 12 \
  --use_faiss \
  --roll_epochs 1 \
  --q_thr 0.45 \
  --model TASER
