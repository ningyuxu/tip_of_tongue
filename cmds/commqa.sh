# --model=llama2_13b,llama2_7b,llama_13b,llama_7b,pythia_1b4,pythia_2b8,pythia_6b9,pythia_12b,falcon_rw_1b,falcon_rw_7b,falcon_7b,phi_1_5,phi_2,mistral_7b,mpt_7b \
# --model=gpt2_xl,gpt_j_6b,bloom_1b7,bloom_3b,bloom_7b1,opt_1b3,opt_2b7,opt_6b7,opt_13b,olmo_1b,olmo_7b
# --corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \

# common sense question answers
# 1) natural language
python -m exps.run commqa --lab=natural --func=probability \
--model=llama2_13b \
--corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \
--nproc=1 --runs=1 --demos=0 --queries=0 --simple=False --gid=0
# 1.1) results
python -m exps.run commqa --lab=natural --func=accuracy \
--model=llama2_13b \
--corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \
--nproc=0 --runs=1 --demos=0 --queries=0 --simple=False

# 2) concept demos
python -m exps.run commqa --lab=concept --func=probability \
--model=llama2_13b \
--corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \
--nproc=1 --runs=5 --demos=1,12,24 --queries=0 --simple=True --gid=0
# 2.1) results
python -m exps.run commqa --lab=concept --func=accuracy \
--model=llama2_13b \
--corpus=arce,arcc,hellaswag,piqa,siqa,openbookqa,boolq,csqa \
--nproc=0 --runs=5 --demos=1,12,24 --queries=0 --simple=False