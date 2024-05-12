for seed in 1 2 3 4 5 6 7 8; do
    python main_policy.py \
        --training_params.inference_algo=mlp --cuda_id=0 --seed=$seed
done