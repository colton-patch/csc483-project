import collections
import math
import argparse


class IRSystem:

    def __init__(self, f):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vecorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        self.docWeights = {}

        for line in f:
            tokens = line.lower().split()
            docId = int(tokens[0])
            termFreqs = collections.defaultdict(int)

            # get frequencies
            for token in tokens[1:]:
                if token != "-":
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
            self.docWeights[docId] = collections.defaultdict(int)

            # get normalized weight for each term
            for term in termFreqs:
                self.docWeights[docId][term] = termFreqs[term] / sqrt


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
        for docId in self.docWeights:
            simScores[docId] = 0
            for term in queryWeights:
                simScores[docId] += self.docWeights[docId][term] * queryWeights[term]             

        # sort and get top 10
        result = sorted(simScores, key=simScores.get, reverse=True)[:10]

        return result


def main(corpus):
    ir = IRSystem(open(corpus, encoding="UTF-8"))

    while True:
        query = input('Query: ').strip()
        if query == 'exit':
            break
        results = ir.run_query(query)
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("CORPUS",
                        help="Path to file with the corpus")
    args = parser.parse_args()
    main(args.CORPUS)