# --model=llama2_13b,llama2_7b,llama_13b,llama_7b,pythia_1b4,pythia_2b8,pythia_6b9,pythia_12b,falcon_rw_1b,falcon_rw_7b,falcon_7b,phi_1_5,phi_2,mistral_7b,mpt_7b \
# --model=gpt2_xl,gpt_j_6b,bloom_1b7,bloom_3b,bloom_7b1,opt_1b3,opt_2b7,opt_6b7,opt_13b,olmo_1b,olmo_7b
# --corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \

# protoqa
# 1) sample protoqa answers (natural language, 0-shot)
python -m exps.run protoqa --lab=sample_answers --task=natural \
--model=llama2_13b \
--greedy=False --temperature=0.7 --top_k=0 --top_p=0.95 --penalty=1.1 \
--nproc=1 --runs=1 --samples=100 --demos=0 --queries=0 --split=dev --gid=0

# 2) sample protoqa answers (concept demos)
python -m exps.run protoqa --lab=sample_answers --task=concept \
--model=llama2_13b \
--greedy=False --temperature=1.0 --top_k=0 --top_p=1.0 --penalty=1.0 \
--nproc=1 --runs=1 --samples=100 --demos=1,12,24 --queries=0 --split=dev --gid=0