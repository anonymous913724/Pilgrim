#!/usr/bin/env bash

python3 soft_topk_attn/yelp/yelp_full_faiss.py \
    --pt data/yelp/yelp_hetero_monthly_C24.pt \
    --use_faiss \
    --quantile 0.25 \
    --model TASER

python3 soft_topk_attn/yelp/yelp_baseline.py \
    --pt data/yelp/yelp_hetero_monthly_C24.pt \
    --quantile 0.25 \
    --model TASER

python3 soft_topk_attn/yelp/yelp_full_faiss.py \
    --pt data/yelp/yelp_hetero_monthly_C24.pt \
    --use_faiss \
    --quantile 0.25 \
    --model SEHTGNN

python3 soft_topk_attn/yelp/yelp_baseline.py \
    --pt data/yelp/yelp_hetero_monthly_C24.pt \
    --quantile 0.25 \
    --model SEHTGNN



