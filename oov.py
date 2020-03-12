import numpy as np


class OOV:
    def __init__(self, words, embeddings, mask_indices):
        self.polyglot_words = words
        self.polyglot_embeddings = embeddings
        self.mask_indices = mask_indices

        # indexes for the corpus embedding
        self.word2idx = {w: i for (i, w) in enumerate(self.polyglot_words)}


    def levenshtein(self, word1, word2):

        def dp(idx1, idx2):
            # insert all idx2 chars
            if idx1 < 0:   
                return idx2 + 1
            
            # delete all idx1 chars
            if idx2 < 0:
                return idx1 + 1
            
            # look in cache
            if (idx1, idx2) in cache:
                return cache[(idx1, idx2)]
            
            if word1[idx1] == word2[idx2]:
                cache[(idx1, idx2)] = dp(idx1 - 1, idx2 - 1)
            else:
                # usual levenshtein
                subs = 1 + dp(idx1 - 1, idx2 - 1)
                insert = 1 + dp(idx1, idx2 - 1) 
                delete = 1 + dp(idx1 - 1, idx2)
                cache[(idx1, idx2)] = min(subs, insert, delete)

                # transposition
                if idx1 > 1 and idx2 > 1 and word1[idx1] == word2[idx2-1] and word1[idx1-1] == word2[idx2]:
                    cache[(idx1, idx2)] = min(cache[(idx1, idx2)], cache[(idx1-2, idx2-2)] + 1)
            
            return cache[(idx1, idx2)]
        
        cache = {}
        return dp(len(word1) - 1, len(word2) - 1)


    def get_closest_neighbor(self, word):

        # use the nearest embedding from polyglot vocabulary
        if word in self.polyglot_words:
            vec = self.polyglot_embeddings[self.word2idx[word]]
            return self.get_closest_embedding_word(vec)
        
        else:
        # find closest levenshtein distance    

            # compute distances
            distances = np.array([self.levenshtein(word, w) for w in self.polyglot_words])
                     
            idx = np.argmin(distances)
            vec = self.polyglot_embeddings[idx]

            return self.get_closest_embedding_word(vec)


    def get_closest_embedding_word(self, vec):
        corpus_embeddings = self.polyglot_embeddings[self.mask_indices]

        cossine_similarities = self._get_cossine_similarities(corpus_embeddings, vec)

        corpus_words = np.array(self.polyglot_words)[self.mask_indices].tolist()
        candidate_idx = np.argmax(cossine_similarities)
        return corpus_words[candidate_idx]


    def _get_cossine_similarities(self, embeddings, vec):
        inners = np.inner(embeddings, vec)
        vec_norm = np.linalg.norm(vec)
        embedding_norms = np.linalg.norm(embeddings)

        return inners / (vec_norm * embedding_norms)
