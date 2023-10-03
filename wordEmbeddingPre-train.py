import sys
import os
from gensim.models import word2vec
from hydra import initialize_config_dir, compose

config_path = os.path.dirname(__file__)

def get_all_formula(txt, max_seq_length):
    all_formulae = list(open(txt, "r", newline="\n"))
    ret = []
    for s in all_formulae:
        formula = ("[BOS] " + s.rstrip("\n") + " [EOS]").split(" ")
        for _ in range(len(formula), max_seq_length):
            formula.append("[PAD]")
        ret.append(formula)
    return ret

def main():
    save_path = sys.argv[1]
    initialize_config_dir(config_dir=config_path, version_base=None)
    config = compose(config_name="config.yaml")
    all_formulae = get_all_formula(config.dataset.all_formulae_lst, config.dataset.max_seq_length) + [["[UNK]"]]
    model = word2vec.Word2Vec(sentences=all_formulae, vector_size=config.model.dim_emb, min_count=0)
    model.wv.save_word2vec_format(os.path.join(save_path, 'embedding.bin'), binary=True)

if __name__ == "__main__":
    main()
