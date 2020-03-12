from collections import defaultdict
import numpy as np
from nltk import Tree
from nltk import Nonterminal
from nltk import induce_pcfg
from nltk import word_tokenize
from PYEVALB import scorer
from PYEVALB import parser


class PCFG:
    def __init__(self, corpus):
        pass


    def train(self, corpus):
        self.corpus = self.preprocess_data(corpus)
        self.trees = self.create_trees(self.corpus)
        self.grammar = self.create_pcfg(self.trees)
        self.vocab = self.get_vocabulary(self.trees)
        self.postag_prob, self.unary_prob, self.binary_prob, self.nonterminals = self.create_dictionaries(self.grammar)


    def remove_functional_labels(self, sentence):
        s = sentence.split(' ')
        for i in range(1, len(s)):
            # check if non-terminal node
            if s[i][0] == '(':
                s[i] = s[i].split('-')[0]
                
        return ' '.join(s)


    def preprocess_data(self, corpus):
        return [self.remove_functional_labels(sentence) for sentence in corpus if sentence != ''] 


    def create_trees(self, corpus):
        trees = [Tree.fromstring(sentence, remove_empty_top_bracketing=True) for sentence in corpus]
        return trees
    

    def create_pcfg(self, trees):
        productions = []
        for tree in trees:
            tree.collapse_unary(collapsePOS=True)
            tree.chomsky_normal_form(horzMarkov=2)
            productions += tree.productions()
        
        S = Nonterminal('SENT')
        grammar = induce_pcfg(S, productions)
        
        return grammar


    def get_vocabulary(self, trees):
        vocab = set()
        for t in trees:
            for word in t.leaves():
                vocab.add(word)

        return vocab


    def create_dictionaries(self, grammar):
        postag_prob = {}
        unary_prob = {}
        binary_prob = {}
        nonterminals = set()

        for prod in grammar.productions():
            nonterminals.add(prod._lhs._symbol)
            
            if prod.is_lexical():
                word = prod._rhs[0]
                pos = prod._lhs._symbol
                
                if word not in postag_prob:
                    postag_prob[word] = {}
                
                postag_prob[word][pos] = prod.prob()
             
            else:
                # add non-terminal tokens
                for A in prod._rhs:
                    nonterminals.add(A._symbol)
                
                # unary transition A -> B
                if len(prod._rhs) == 1:
                    A = prod._lhs._symbol
                    B = prod._rhs[0]._symbol
                    if B not in unary_prob:
                        unary_prob[B] = {}
                    unary_prob[B][A] = prod.prob()

                if len(prod._rhs) == 2:
                    A = prod._lhs._symbol
                    B, C = prod._rhs[0]._symbol, prod._rhs[1]._symbol

                    if (B, C) not in binary_prob:
                        binary_prob[(B, C)] = {}
                    binary_prob[(B, C)][A] = prod.prob()


        return postag_prob, unary_prob, binary_prob, nonterminals


    def CYK(self, words, grammar):
        
        scores = defaultdict(dict)
        backpointers = {}
        replace_words = {}

        def handle_unaries(begin, end):
            added = True
            while added:
                added = False

                for B in self.unary_prob.keys():
                    for A in self.unary_prob[B]:
                        
                        if B in scores[(begin, end)]:
                            prob = self.unary_prob[B][A] * scores[(begin, end)][B]
                            if A not in scores[(begin, end)] or prob > scores[(begin, end)][B]:
                                scores[(begin, end)][A] = prob
                                backpointers[(begin, end, A)] = [(begin, end, B)]
                                added = True


        # initialize table with terminal tokens
        for i in range(len(words)):
            w = words[i]
            
            # handle oov words
            if w not in self.vocab:
                w = self.oov.get_closest_neighbor(w)               
                replace_words[(i, w)] = words[i]
                
            for pos in self.postag_prob[w].keys():
                scores[(i, i+1)][pos] = self.postag_prob[w][pos]
                backpointers[(i, i+1, pos)] = [(i, i+1, w)]

            handle_unaries(i, i+1)
        
        # dynamic-programming in the parser diamond
        for span in range(2, len(words)+1):
            for begin in range(len(words) + 1 - span):
                end = begin + span
                for split in range(begin+1, end):

                    for B in scores[(begin, split)].keys():
                        for C in scores[(split, end)].keys():
                            if (B, C) in self.binary_prob:
                                for A in self.binary_prob[(B, C)].keys():
                                    prob = scores[(begin, split)][B] * scores[(split, end)][C] * self.binary_prob[(B, C)][A]
                                    if A not in scores[(begin, end)] or prob > scores[(begin, end)][A]:
                                        scores[(begin, end)][A] = prob
                                        backpointers[(begin, end, A)] = [(begin, split, B), (split, end, C)]

                handle_unaries(begin, end)


        return scores, backpointers, replace_words


    def build_tree(self, scores, backpointers, words):
        n = len(words)
        if 'SENT' not in scores[(0, n)]:
            # not able to parse the sentence
            return
        else:
            def aux(begin, end, token):

                if len(backpointers[(begin, end, token)]) == 1:
                    tag = backpointers[(begin, end, token)][0][2]
                    # check if it's a word
                    if tag in self.vocab:
                        # return '{}'.format(token)
                        return '({} {})'.format(token, tag)
                    else:
                        tup = backpointers[(begin, end, token)][0]
                        return '({} {})'.format(token, aux(*tup))


                left, right = backpointers[(begin, end, token)]

                return '({} {} {})'.format(token, aux(*left), aux(*right))

            return aux(0, n, 'SENT')


    def generate_output(self, corpus):

        corpus = self.preprocess_data(corpus)
        trees = self.create_trees(corpus)
        output = []
        for i, tree in enumerate(trees):

            predicted = self.parse_tree(tree)
            
            if predicted == '<EMTPY>':
                output.append('(SENT (UNK))')
            else:
                output.append(' '.join(predicted.split()))              
                
            if i % 100 == 0:
                print(i)

        with open('evaluation_data.parser_output', 'w') as f:
            f.write('\n'.join(output))

        return output


    def parse_from_txt(self, corpus):
        output = []
        for sentence in corpus[:-1]:
            predicted = self.parse_sentence(sentence)

            if predicted == '<EMTPY>':
                output.append('(SENT (UNK))')
            else:
                output.append(' '.join(predicted.split()))               

        with open('output', 'w') as f:
            f.write('\n'.join(output))

        return output


    def predict(self, corpus):

        corpus = self.preprocess_data(corpus)
        trees = self.create_trees(corpus)

        accs = []
        nb_no_parse = 0

        for i, tree in enumerate(trees):

            target = self.preprocess_target(corpus[i])
            predicted = self.parse_tree(tree)

            if predicted == '<EMTPY>':
                nb_no_parse += 1
            else:              
                acc = self.get_accuracy(target, predicted)
                accs.append(acc)

            if i % 100 == 0:
                print(i)

        print('not able to parse: {}'.format(nb_no_parse))
        mean_acc = sum(accs) / len(accs)
        print('mean acc: {}'.format(mean_acc))

        return accs, nb_no_parse


    def preprocess_target(self, target):
        target = target[1:-1]
        tree = Tree.fromstring(target)
        tree.collapse_unary(collapsePOS=True)
        target = ' '.join(str(tree).split())        
        return target


    def get_accuracy(self, target, predicted):
        gold_tree = parser.create_from_bracket_string(target)
        test_tree = parser.create_from_bracket_string(predicted)

        s1 = np.array(gold_tree.poss)
        s2 = np.array(test_tree.poss)

        acc = np.sum(s1 == s2) / s1.shape[0]
        return acc

  
    def parse_sentence(self, sentence):
        sentence = word_tokenize(sentence)
        # apply the CYK algorithm to a string and return the parse tree
        scores, backpointers, replace_words = self.CYK(sentence, self.grammar)
        parse = self.build_tree(scores, backpointers, sentence)
        if parse:
            # post-process the tree to undo chomsky normal form
            tree = Tree.fromstring(parse)
            tree.un_chomsky_normal_form()
            return str(tree)
        else:
            return '<EMTPY>'


    def parse_tree(self, tree):
        # preprocess tree to chomsky normal form
        tree.collapse_unary(collapsePOS=True)
        tree.chomsky_normal_form(horzMarkov=2)

        # apply CYK algorithm and build the parse tree
        scores, backpointers, replace_words = self.CYK(tree.leaves(), self.grammar)
        parse = self.build_tree(scores, backpointers, tree.leaves())

        if parse:
            # post-process the tree to undo chomsky normal form
            tree = Tree.fromstring(parse)
            tree.un_chomsky_normal_form()
            return str(tree)
        else:
            return '<EMTPY>'


    def set_oov(self, oov, words, embeddings):
        # mask to get word embeddings of the training corpus
        intersection = self.vocab.intersection(set(words))
        word2idx = {w: i for (i, w) in enumerate(words)}
        mask_indices = [word2idx[word] for word in intersection]

        self.oov = oov(words, embeddings, mask_indices)
