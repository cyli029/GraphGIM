python finetune.py \
    --dataset 'bbbp' \
    --lr 1e-5 \
    --epochs 100 \
    --dropout 0.5 \
    --emb_dim 300 \
    --resume 'best_epoch=1_loss=2.62' \
    --pretrained \