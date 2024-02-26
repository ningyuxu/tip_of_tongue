# --model=llama2_13b,llama2_7b,llama_13b,llama_7b,pythia_1b4,pythia_2b8,pythia_6b9,pythia_12b,falcon_rw_1b,falcon_rw_7b,falcon_7b,phi_1_5,phi_2,mistral_7b,mpt_7b \

# 1) concept inference
python -m exps.run concept --lab=inference --func=embed_concept \
--model=llama2_7b \
--task=d2w --nproc=1 --runs=1 --demos=1 --queries=0 --gid=0

# 2) check generation results
python -m exps.run concept --lab=inference  --func=gener_results \
--model=llama2_7b \
--task=d2w --nproc=0 --demos=1 --queries=0

# 3) check exact match results
python -m exps.run concept --lab=inference  --func=match_results \
--model=llama2_7b \
--task=d2w --nproc=0 --runs=1 --demos=1 --queries=0

# 4) clone concept
python -m exps.run concept --lab=inference --func=clone_concept \
--model=falcon_7b --base_model=llama2_7b \
--task=d2w --nproc=1 --demos=1 --runs=1 --queries=0 --gid=0
# 4.1) check ld2w results
python -m exps.run concept --lab=inference  --func=match_results \
--model=falcon_7b \
--task=d2w --nproc=0 --runs=1 --demos=0 --queries=0

# 5) ld2w task
python -m exps.run concept --lab=inference --func=embed_concept \
--model=llama2_7b \
--task=ld2w --nproc=1 -runs=1 --demos=0 --queries=0 --gid=0
# 5.1) check ld2w results
python -m exps.run concept --lab=inference  --func=match_results \
--model=llama2_7b \
--task=ld2w --nproc=0 --runs=1 --demos=0 --queries=0

# 6) baseline
python -m exps.run concept --lab=inference --func=baseline --baseline=shuffle,mismatch,random \
--model=llama2_7b \
--task=d2w --nproc=1 --runs=1 --demos=1 --queries=0 --gid=0
# 6.1) check baseline results
python -m exps.run concept --lab=inference  --func=match_results \
--model=llama2_7b \
--task=d2w --nproc=0 --runs=1 --demos=1 --queries=0 --baseline=shuffle

# 7) description influence
python -m exps.run concept --lab=influence --func=description --descs=things,wordnet,hill200,shuffle \
--model=llama2_7b \
--task=d2w --nproc=1 --runs=1 --demos=1 --queries=0 --gid=0
# 7.1) show influence description results
python -m exps.run concept --lab=influence --func=desc_influ_result --descs=things,wordnet,hill200,shuffle \
--model=llama2_7b \
--task=d2w --nproc=0 --runs=1 --demos=1 --queries=0

# 8) wordnet influence
python -m exps.run concept --lab=influence --func=concept \
--model=llama2_7b \
--task=d2w --nproc=1 --runs=1 --demos=1 --queries=0 --gid=0
# 8.1) wordnet results
python -m exps.run concept --lab=influence --func=concept_influ_result \
--model=llama2_7b \
--task=d2w --nproc=0 --runs=1 --demos=1 --queries=0
