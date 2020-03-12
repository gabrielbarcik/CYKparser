import loader
import argparse
from pcfg import PCFG
from oov import OOV

TREEBANK_PATH = 'sequoia-corpus+fct.mrg_strict'
EMBEDDING_PATH = 'polyglot-fr.pkl'


def run(args):
    
    data = loader.load_treebanks(TREEBANK_PATH)
    train_data, dev_data, test_data = loader.train_test_split(data, 0.8, 0.1, 0.1)
    words, embeddings = loader.load_word_embeddings(EMBEDDING_PATH)

    pcfg = PCFG(train_data)
    pcfg.train(train_data)
    pcfg.set_oov(OOV, words, embeddings)

    if args.generate_output:
        output = pcfg.generate_output(test_data)

    if args.evaluation:
        accs, nb_no_parse = pcfg.predict(test_data[:2])

    if args.parse:
        corpus = []
        with open(args.txt_path, 'r') as f:
            corpus = f.read().split('\n')
        pcfg.parse_from_txt(corpus)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='PCFG parser using the CYK algorithm')

    argparser.add_argument('--generate_output', action='store_true', help='generate evaluation_data.parser_output file on test data')
    argparser.add_argument('--evaluation', action='store_true', help='performs POS evaluation on test data')
    argparser.add_argument('--parse', action='store_true', help='parse .txt file and save on output file')
    argparser.add_argument('txt_path', nargs="?", help='path for txt file to be parsed, used if parse flag is True')

    run(argparser.parse_args())
    import pdb; pdb.set_trace()
