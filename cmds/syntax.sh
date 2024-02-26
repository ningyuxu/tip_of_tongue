# models
# --model=llama2_13b,llama2_7b,llama_13b,llama_7b,pythia_1b4,pythia_2b8,pythia_6b9,pythia_12b,falcon_rw_1b,falcon_rw_7b,falcon_7b,phi_1_5,phi_2,mistral_7b,mpt_7b \


# blimp
# 1) calculate probability
python -m exps.run syntax --lab=blimp --func=probability \
--model=llama2_13b \
--nproc=1 --gid=0

# 2) calculate score
python -m exps.run syntax --lab=blimp --func=score \
--model=llama2_13b \
--nproc=0


# syntaxgym
# 1) calculate surprisal
python -m exps.run syntax --lab=syntaxgym --func=surprisal \
--model=llama2_13b \
--nproc=1 --gid=0

# 2) calculate score
python -m exps.run syntax --lab=syntaxgym --func=score \
--model=llama2_13b \
--nproc=0
