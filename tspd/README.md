# Quick Demo (20 nodes, short training)
python main.py --mode demo --n_nodes 20 --epochs 500

# N=20, Random dataset
python main.py \
    --mode train \
    --n_nodes 20 \
    --epochs 10000 \
    --batch_size 128 \
    --d_model 128 \
    --n_encoder_layers 3 \
    --lr 1e-4 \
    --device cuda \
    --instance_type random \
    --model_path model_n20.pt

# N=50
python main.py --mode train --n_nodes 50 --epochs 10000 \
    --batch_size 128 --device cuda --model_path model_n50.pt

# N=100
python main.py --mode train --n_nodes 100 --epochs 10000 \
    --batch_size 64 --device cuda --model_path model_n100.pt

# Evaluation (sampling strategies)
python main.py \
    --mode eval \
    --n_nodes 20 \
    --model_path model_n20.pt \
    --n_samples 4800 \
    --device cuda