
import collections
import math
import os
import re

class IRSystem:
    def __init__(self, files):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vectorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        self.docWeights = {}

        for filename in files:
            with open(os.path.join("./wiki-subset-20140602", filename), encoding="UTF-8") as f:
                docName = None
                tokens = []
                for line in f:
                    line = line.strip()
                    # at the end of the article, process its weights
                    if line.startswith("[[") and line.endswith("]]"):
                        if docName is not None and tokens:
                            self._get_weights(docName, tokens)
                        # start new article, denoted by "[[article name]]"
                        docName = line[2:-2]
                        tokens = []

                    # add tokenized line to tokens
                    lowerLine = line.lower()
                    tokens += re.sub(r'[^a-zA-Z0-9\s]', '', lowerLine).split()
                

                # process last article
                if docName is not None and tokens:
                    self._get_weights(docName, tokens) 
    def _get_weights(self, docName, tokens):
        # get frequencies
        termFreqs = collections.defaultdict(int)
        for token in tokens:
            if token:
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
        lowerQuery = query.lower()
        terms = re.sub(r'[^a-zA-Z0-9\s]', '', lowerQuery).split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-50 documents for the query 'terms'

        termFreqs = collections.defaultdict(int)

        # count term freqs
        for term in terms:
            if term:
                termFreqs[term] += 1

        # replace term frequencies with logarithmic term frequencies
        for term in termFreqs:
            termFreqs[term] = 1 + math.log10(termFreqs[term])

        # get df
        docFreqs = collections.defaultdict(int)
        for term in termFreqs:
            for doc in self.docWeights:
                if term in self.docWeights[doc]:
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

        # sort and get top 50
        result = sorted(simScores, key=simScores.get, reverse=True)[:100]

        return result

def send_to_llm(query, results, expected_answer):
    # Placeholder function to simulate sending query and results to an LLM and getting a response.
    # In a real implementation, this function would interact with an LLM API.
    response_from_llm = results[0] # Simulating that LLM returns the top result as the answer.
    return response_from_llm == expected_answer.lower()

def main():
    ir = IRSystem(os.listdir("./wiki-subset-20140602"))

    counter = 0

    with open('questions.txt', 'r') as f:
        lines = [line.strip().lower() for line in f.readlines()[:3]]
 
    query = ' '.join(re.split(r'\W+', lines[0] + ' ' + lines[1]))
    results = ir.run_query(query)
    correct = send_to_llm(query, results, lines[2])

    if correct:
        counter += 1

    print(f'Counter of Correct Answers: {counter}')


if __name__ == '__main__':
    main()
