# check huggingface models
python -m exps.run setup --component=tokenizer

# check huggingface tokenizers
python -m exps.run setup --component=model

# check corpus
python -m exps.run corpus --name=concrete
python -m exps.run corpus --name=things
python -m exps.run corpus --name=revdict
python -m exps.run corpus --name=wordnet --func=tree
python -m exps.run corpus --name=wordnet --func=bc5000
python -m exps.run corpus --name=wordnet --func=things
python -m exps.run corpus --name=wordnet --func=word
python -m exps.run corpus --name=wordnet
python -m exps.run corpus --name=protoqa
python -m exps.run corpus --name=arce
python -m exps.run corpus --name=arcc
python -m exps.run corpus --name=hellaswag
python -m exps.run corpus --name=piqa
python -m exps.run corpus --name=siqa
python -m exps.run corpus --name=openbookqa
python -m exps.run corpus --name=boolq
python -m exps.run corpus --name=csqa
python -m exps.run corpus --name=blimp
python -m exps.run corpus --name=sg_eval
python -m exps.run corpus --name=cslb