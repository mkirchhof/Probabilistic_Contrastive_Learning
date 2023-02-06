python main.py \
--g_dim_z 10 \
--g_dim_x 10 \
--e_dim_z 32 \
--g_pos_kappa 20 \
--g_post_kappa_min 16 \
--g_post_kappa_max 32 \
--l_n_samples 512 \
--bs 512 \
--n_neg 8 \
--n_batches_per_half_phase 50000 \
--use_wandb False \
--loss MCInfoNCE \
--l_learnable_params False \
--n_phases 1 \
--seed 8
