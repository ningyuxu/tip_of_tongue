import re
import shutil
from typing import List, Iterable, Dict
from calf import logger, CORPUS, CACHE
from .. import enumeration as enum
from ..iso639 import ISO639
from ..file import download, extract
from .corpus import Corpus

RE_SENT_ID = re.compile(r"^# sent_id\s*=?\s*(\S+)")
RE_TEXT = re.compile(r"^# text\s*=\s*(.*)")
RE_NEWPARDOC = re.compile(r"^# (newpar|newdoc)(?:\s+id\s*=\s*(.+))?$")


class UDTreebankCorpus(Corpus):

    corpus = "ud_treebank"
    version = "v2.10"
    fields = ["doc_id", "par_id", "sent_id", "text", "id", "form", "lemma", "upos", "xpos", "feats",
              "head", "deprel", "deps", "misc"]
    corpus_path = CORPUS / f"{corpus}_{version}"
    corpus_filename = f"{corpus}_{version}.tgz"

    upos_vocab = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
                  "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    deprel_vocab = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf",
                    "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated",
                    "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj",
                    "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
                    "vocative", "xcomp"]

    def __init__(self,
                 url: str =
                 "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/"
                 "ud-treebanks-v2.10.tgz",
                 lang: str = "en",
                 genre: str = "EWT",
                 split: str = enum.Split.TRAIN,
                 reload: bool = False) -> None:
        super().__init__(reload=reload, name=self.corpus, lang=lang, genre=genre, split=split)
        self.url = url
        self.lang_code = lang
        self.lang_name = ISO639.match(self.lang_code).name
        self.genre = genre
        self.split = split
        self.corpus_file_path = self.corpus_path / f"UD_{self.lang_name}-{self.genre}"

    def is_initialized(self) -> bool:
        return self.corpus_path.is_dir() and any(self.corpus_path.iterdir())

    def init(self, reload: bool = False, clean: bool = False) -> None:
        # reset corpus
        self.reset()
        # download corpus
        udtreebank_file = CACHE / self.corpus_filename
        if not udtreebank_file.is_file():
            logger.info(f"Downloading {self.corpus} corpus ...")
            download(url=self.url, file=udtreebank_file)
            logger.info(f"Done.")
        logger.info(f"Extracting {self.corpus} corpus ...")
        extract(compressed_file=udtreebank_file, destination_folder=self.corpus_path, clean=False)
        logger.info(f"Done")

    def reset(self) -> None:
        shutil.rmtree(self.corpus_path, ignore_errors=True)

    def load(self) -> Iterable:
        annotations = {"doc_id": None, "par_id": None, "sent_id": None, "text": None}
        word_lines = []
        with open(self.get_file(), mode='r', encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    yield self.parse_lines(annotations, word_lines)
                    word_lines = []
                else:
                    if line.startswith('#'):
                        sent_id_match = RE_SENT_ID.match(line)
                        if sent_id_match is not None:
                            annotations["sent_id"] = sent_id_match.group(1)
                        text_match = RE_TEXT.match(line)
                        if text_match is not None:
                            annotations["text"] = text_match.group(1)
                        pardoc_match = RE_NEWPARDOC.match(line)
                        if pardoc_match is not None:
                            value = True if pardoc_match.group(2) is None else pardoc_match.group(2)
                            if pardoc_match.group(1) == "newpar":
                                annotations["par_id"] = value
                            else:
                                annotations["doc_id"] = value
                    else:
                        word_lines.append(line)
            if word_lines:
                yield self.parse_lines(annotations, word_lines)

    @staticmethod
    def parse_lines(annotations: Dict, conll_lines: Iterable[str]) -> List:
        doc_id = annotations.get("doc_id", None)
        par_id = annotations.get("par_id", None)
        sent_id = annotations.get("sent_id", None)
        text = annotations.get("text", None)
        # if doc_id is not None and par_id is not None:
        #     d_id, par_id = par_id.rsplit('-', 1)
        #     assert d_id == doc_id
        # if doc_id is not None and sent_id is not None:
        #     d_id, sent_id = sent_id.rsplit('-', 1)
        #     assert d_id == doc_id
        sentence = []
        for line in conll_lines:
            conll_tags: List = line.strip().split('\t')
            if not conll_tags[0].isdigit():
                continue
            else:
                conll_tags[0] = int(conll_tags[0])  # id
                conll_tags[6] = int(conll_tags[6])  # head
                conll_tags[7] = conll_tags[7].split(':')[0]
                sentence.append(conll_tags)
        return [doc_id, par_id, sent_id, text] + list(zip(*sentence))

    def get_file(self) -> str:
        file_name = f"{self.lang_code}_{self.genre.lower()}-ud-{self.split}.conllu"
        return str(self.corpus_file_path / file_name)
