import collections
import math
import os
import re


class IRSystem:

    def __init__(self, files):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vecorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        self.docWeights = {}

        for filename in files:
            with open(os.path.join("./wiki-subset-20140602", filename), encoding="UTF-8") as f:
                docName = None
                tokens = []

                for line in f:
                    line = line.strip()

                    # at the end of the article, process it's weights
                    if line.startswith("[[") and line.endswith("]]"):
                        if docName is not None and tokens:
                            self._get_weights(docName, tokens)

                        # start new article, denoted by "[[article name]]"
                        docName = line[2:-2]
                        tokens = []

                    else:
                        # add tokenized line to tokens
                        lowerLine = line.lower()
                        tokens += re.split(r'[^a-zA-Z0-9]+', lowerLine)
                
                # process last article
                if docName is not None and tokens:
                    self._get_weights(docName, tokens)
                
    def _get_weights(self, docName, tokens):
        # get frequencies
        termFreqs = collections.defaultdict(int)
        for token in tokens:
            if (token not in termFreqs):
                termFreqs[token] = 0
            termFreqs[token] += 1

        sumSquares = 0
        
        for term in termFreqs:
            # replace the values in termFreqs with the logarithmic vals
            termFreqs[term] = 1 + math.log10(termFreqs[term])
            # add the square of the logTf to get vector
            sumSquares += termFreqs[term] ** 2

        sqrt = math.sqrt(sumSquares)

        # make the entry for each document a dict with terms as keys
        self.docWeights[docName] = collections.defaultdict(int)

        # get normalized weight for each term
        for term in termFreqs:
            self.docWeights[docName][term] = termFreqs[term] / sqrt


    def run_query(self, query):
        terms = query.lower().split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'

        # YOUR CODE GOES HERE
        termFreqs = {}
        
        # count term freqs
        for term in terms:
            if term not in termFreqs:
                termFreqs[term] = 0
            termFreqs[term] += 1

        # replace term frequencies with logarithmic term frequencies
        for term in termFreqs:
            termFreqs[term] = 1 + math.log10(termFreqs[term])

        # get df
        docFreqs = collections.defaultdict(int)

        for term in termFreqs:
            for doc in self.docWeights:
                if term in self.docWeights[doc] and self.docWeights[doc][term] != 0:
                    if term not in docFreqs:
                        docFreqs[term] = 0
                    docFreqs[term] += 1

        # replace dfs with idfs
        n = len(self.docWeights)
        for term in docFreqs:
            docFreqs[term] = math.log10(n / docFreqs[term])

        # calculate query weights
        queryWeights = {}
        for term in termFreqs:
            queryWeights[term] = termFreqs[term] * docFreqs[term]

        # get similarity score
        simScores = {}
        for docName in self.docWeights:
            simScores[docName] = 0
            for term in queryWeights:
                simScores[docName] += self.docWeights[docName][term] * queryWeights[term]            

        # sort and get top 10
        result = sorted(simScores, key=simScores.get, reverse=True)[:10]

        return result

# this is my branch
def main():
    ir = IRSystem(os.listdir("./wiki-subset-20140602"))

    while True:
        query = input('Query: ').strip()
        if query == 'exit':
            break
        results = ir.run_query(query)
        print(results)


if __name__ == '__main__':
    main()