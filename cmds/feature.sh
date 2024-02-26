# common sense question answers
# 1) natural language
# --model=llama2_13b,llama2_7b,llama_13b,llama_7b,pythia_1b4,pythia_2b8,pythia_6b9,pythia_12b,falcon_rw_1b,falcon_rw_7b,falcon_7b,phi_1_5,phi_2,mistral_7b,mpt_7b \
# --model=gpt2_xl,gpt_j_6b,bloom_1b7,bloom_3b,bloom_7b1,opt_1b3,opt_2b7,opt_6b7,opt_13b,olmo_1b,olmo_7b
# --corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \
python -m exps.meaning.run feature --nproc=1 --runs=1 \
--model=llama2_7b \
--lab=probability --task=concept --corpus=selected --demos=0 --things_demos=1,12,24,48 --queries=1000 --gid=0

python -m exps.meaning.run feature --nproc=1 --runs=1 \
--model=llama2_7b \
--lab=probability --task=natural --corpus=selected --demos=0 --things_demos=0 --queries=1000 --gid=0

python -m exps.meaning.run feature --nproc=1 --runs=1 \
--model=llama2_7b \
--lab=probability --task=description --corpus=selected --demos=0 --things_demos=0 --queries=1000 --gid=0

# -- score
python -m exps.meaning.run feature --nproc=0 --runs=1 \
--model=llama2_7b \
--lab=score --task=concept --corpus=selected --demos=0 --things_demos=1 --queries=0

python -m exps.meaning.run feature --nproc=0 --runs=1 \
--model=llama2_7b \
--lab=score --task=natural --corpus=selected --demos=0 --things_demos=0 --queries=1000

python -m exps.meaning.run feature --nproc=0 --runs=1 --demos=0 \
--model=llama2_7b \
--lab=score --task=description --corpus=selected --things_demos=0 --queries=1000