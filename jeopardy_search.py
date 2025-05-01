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

        # Store the vectorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        self.docWeights = {}
        self.docFreqs = collections.defaultdict(int)

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
                    curTokens = re.sub(r'[^a-zA-Z0-9\s]', '', lowerLine).split()
                    for token in curTokens:
                        tokens.append(token)

                        # count doc frequency of tokens
                        if token not in self.docFreqs:
                            self.docFreqs[token] = 0
                        self.docFreqs[token] += 1    

                # process last article
                if docName is not None and tokens:
                    self._get_weights(docName, tokens) 


    def _get_weights(self, docName, tokens):
        # get frequencies
        termFreqs = collections.defaultdict(int)
        for token in tokens:
            if token:
                termFreqs[token] += 1

                # make the entry for each term a dict with docNames as keys and weights as values
                if token not in self.docWeights:
                    self.docWeights[token] = collections.defaultdict(int)

        sumSquares = 0

        for term in termFreqs:
            # replace the values in termFreqs with the logarithmic vals
            termFreqs[term] = 1 + math.log10(termFreqs[term])
            # add the square of the logTf to get vector
            sumSquares += termFreqs[term] ** 2

        sqrt = math.sqrt(sumSquares)

        # get normalized weight for each term
        for term in termFreqs:
            self.docWeights[term][docName] = termFreqs[term] / sqrt


    def run_query(self, query):
        lowerQuery = query.lower()
        terms = re.sub(r'[^a-zA-Z0-9\s]', '', lowerQuery).split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-50 documents for the query 'terms'

        termFreqs = collections.defaultdict(int)

        # count term freqs
        for term in terms:
            if term:
                termFreqs[term] += 1

        # replace term frequencies with logarithmic term frequencies
        for term in termFreqs:
            termFreqs[term] = 1 + math.log10(termFreqs[term])

        # calculate idfs and query weights
        n = len(self.docWeights)
        queryWeights = {}
        for term in termFreqs:
            if self.docFreqs[term] != 0:
                idf = math.log10(n / self.docFreqs[term])
            else:
                idf = 0
            queryWeights[term] = termFreqs[term] * idf

        # get similarity score
        simScores = collections.defaultdict(int)
        for term in termFreqs:
            if queryWeights[term] > 0:
                for docName in self.docWeights[term]:
                    simScores[docName] += self.docWeights[term][docName] * queryWeights[term] 

        # sort and get top 50
        result = sorted(simScores, key=simScores.get, reverse=True)[:1000]
        return result

      def calculate_mrr(self, results, correct_answers):
        reciprocal_rank_sum = 0
        for answer in correct_answers:
            rank = next((i + 1 for i, result in enumerate(results) if result == answer), 0)
            if rank > 0:
                reciprocal_rank_sum += 1 / rank
        return reciprocal_rank_sum / len(correct_answers)

def main():
    ir = IRSystem(os.listdir("./wiki-subset-20140602"))

    counter = 0
    mrr_sum = 0

    with open('questions.txt', 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 4):
        if i + 2 >= len(lines):
            break
        queryLine = [lines[i].strip().lower(), lines[i+1].strip().lower()]
        query = re.sub(r'[^a-zA-Z0-9\s]', '', queryLine[0] +' '+queryLine[1])
        # print(query)
        results = ir.run_query(query)

        # check against answers
        answer_line = lines[i+2].strip()
        if '|' in answer_line:
            answers = answer_line.split('|')
        else:
            answers = [answer_line]
        
        counter += sum(1 for answer in answers if answer in results)
        mrr_sum += ir.calculate_mrr(results, answers)

    print(f'Counter of Correct Answers: {counter}')
    print(f'Mean Reciprocal Rank (MRR): {mrr_sum / (100)}')

if __name__ == '__main__':
    main()
