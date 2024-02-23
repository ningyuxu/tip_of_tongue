# Conceptual Representation in Large Language Models

This repository contains code for the paper [_On the Tip of the Tongue: Analyzing Conceptual Representation in Large Language Models with Reverse-Dictionary Probe_](https://arxiv.org/abs/2402.14404).



The table below shows the performance of different LLMs in the reverse dictionary task when provided with 24 description–word pairs (averaged across five runs).

We incorporate some newly launched LLMs for comparison ([OLMo](https://arxiv.org/abs/2402.00838) by AllenAi and [Gemma](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) by Google).

| Model        | Demo 24 |
|--------------|---------|
| pythia-1b4   | 46.5    |
| pythia-2b8   | 52.4    |
| pythia-6b9   | 60.1    |
| pythia-12b   | 63.8    |
| phi-1.5      | 52.1    |
| phi-2        | 65.5    |
| falcon-rw-1b | 51.9    |
| falcon-rw-7b | 67.8    |
| falcon-7b    | 72.5    |
| mpt-7b       | 70.9    |
| llama-7b     | 70.9    |
| llama-13b    | 73.8    |
| llama2-7b    | 73.0    |
| llama2-13b   | 78.3    |
| mistral-7b   | 77.6    |
| olmo-1b      | 49.6    |
| olmo-7b      | 69.7    |
| gemma-2b     | 67.0    |
| gemma-7b     | 77.7    |


## Experiments

### Reverse-Dictionary Probe

#### Behavioral Analysis


We use the description–word pairs primarily sourced from the [THINGS](https://osf.io/jum2f/) database (Hebart et al., 2019), which encompasses a broad list of 1,854 concrete and nameable object concepts. We randomly select N word-description pairs as demonstrations and vary N from 1 to 48 to test the impact of the number of demonstrations on LLMs' behavior. 

To test the robustness of LLMs, we further include in our analysis the corresponding descriptions of these objects in [WordNet](https://wordnet.princeton.edu/) (Fellbaum, 1998) and an additional 200 pairs of words and human-written descriptions created by Hill et al. (2016).

We conduct the experiments on 15 open-source Transformer-based large language models (LLMs) pretrained autoregressively for next-word prediction, including (1) the Falcon models, (2) LLaMA models, (3) Mistral 7B, (4) MPT model, (5) Phi models, and (6) the Pythia suite. We use the models accessible through HuggingFace, which are listed in [`config.yaml`](./exps/config.yaml).

Code for this experiment can be found in the [`exps/concept`](./exps/concept) directory.




#### Representation Analysis


For categorization, we use the high-level natural categories from the THINGS database. Following Hebart et al. (2020), we remove subcategories of other categories, concepts belonging to multiple categories and categories with fewer than ten concepts. This results in 18 out of 27 categories comprising 1,112 concepts.

In terms of feature decoding, we use the XCSLB feature norm (Misra et al., 2022) for our analysis, which expands the original CSLB dataset (Devereux et al., 2014) with necessary modifications for consistency. XCSLB includes 3,645 descriptive features for 521 concepts. We take the concepts that overlap with those in THINGS and remove features that are too sparse with fewer than 20 concepts. This results in 257 features associated with 388 concepts in total.

Code is in the [`exps/representation`](./exps/representation) directory.

### Implications of Conceptual Inference

#### Conceptual Inference Ability Predicts Commonsense Reasoning Performance

We conduct a correlation analysis to examine the relationship between conceptual inference (evaluated by the reverse-dictionary task) and the general commonsense reasoning abilities of LLMs.

We take widely-used benchmarks to evaluate LLMs' general knowledge and reasoning ability, including CommonsenseQA (CSQA; Talmor et al., 2019), ARC easy (ARC-E) and challenge (ARCC; Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), Hellaswag (Zellers et al., 2019) and BoolQ (Clark et al., 2019).

Code for this experiment is in [`exps/commqa`](./exps/commqa).

#### Relationship between Conceptual Inference and Syntactic Generalization

We use two benchmarks for evaluating models' syntactic generalization: SyntaxGym (Hu et al., 2020; Gauthier et al., 2020) and the Benchmark of Linguistic Minimal Pairs (BLiMP; Warstadt et al., 2020), which cover a wide range of linguistic phenomena.

Code for this experiment can be found in [`exps/syntax`](./exps/syntax).

#### Generalizing Reverse Dictionary to Commonsense Reasoning

We use the development set of ProtoQA (Boratko et al., 2020) for evaluation as the answers to the test sets are not publicly available. For evaluation, we use the public source code at [https://github.com/iesl/protoqa-evaluator](https://github.com/iesl/protoqa-evaluator).

Code for this experiment is in the [`exps/protoqa`](./exps/protoqa) directory.

## Environment

```
# conda environment
conda create -n tot python=3.11
conda activate tot

# install python library
pip install torch  
pip install jupyter transformers fire omegaconf scipy nltk pandas scikit-learn wordfreq

# prepare corpus
python -c "import nltk; nltk.download('wordnet')"
python -c "import nltk; nltk.download('semcor')"
```

## Citation

```
@article{xu2024tip,
  title={On the Tip of the Tongue: Analyzing Conceptual Representation in Large Language Models with Reverse-Dictionary Probe}, 
  author={Ningyu Xu and Qi Zhang and Menghan Zhang and Peng Qian and Xuanjing Huang},
  year={2024},
  journal={arXiv preprint arXiv:2402.14404},
  url={https://arxiv.org/abs/2402.14404}
}
```


## References

Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. AAAI.

Michael Boratko, Xiang Li, Tim O’Gorman, Rajarshi Das, Dan Le, and Andrew McCallum. 2020. ProtoQA: A question answering dataset for prototypical common-sense reasoning. EMNLP.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. NAACL. 

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv.

Barry J. Devereux, Lorraine K. Tyler, Jeroen Geertzen, and Billi Randall. 2014. The Centre for Speech, Language and the Brain (CSLB) concept property norms. Behavior Research Methods, 46(4):11191127.

Christiane Fellbaum. 1998. WordNet: An electronic lexical database. MIT press.

Jon Gauthier, Jennifer Hu, Ethan Wilcox, Peng Qian, and Roger Levy. 2020. SyntaxGym: An online platform for targeted evaluation of language models. ACL. 

Martin N. Hebart, Adam H. Dickter, Alexis Kidder, Wan Y. Kwok, Anna Corriveau, Caitlin Van Wicklin, and Chris I. Baker. 2019. Things: A database of 1,854 object concepts and more than 26,000 naturalistic object images. PLOS ONE, 14(10):1–24.

Martin N. Hebart, Charles Y. Zheng, Francisco Pereira, and Chris I. Baker. 2020. Revealing the multidimensional mental representations of natural objects underlying human similarity judgements. Nature Human Behaviour, 4(11):1173–1185.

Felix Hill, Kyunghyun Cho, Anna Korhonen, and Yoshua Bengio. 2016. Learning to understand phrases by embedding the dictionary. TACL. 

Jennifer Hu, Jon Gauthier, Peng Qian, Ethan Wilcox, and Roger Levy. 2020. A systematic assessment of syntactic generalization in neural language models. ACL. 

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. EMNLP. 

Kanishka Misra, Julia Rayz, and Allyson Ettinger. 2022. A property induction framework for neural language models. CogSci. 

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense reasoning about social interactions. EMNLP. 

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. NAACL. 

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The benchmark of linguistic minimal pairs for English. TACL.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? ACL.
