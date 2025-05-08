# Drew Marien & Colton Patch Final project Jeopardy
# csc 483 spring 2025
# This code is trying to loop through, parse, and normalize data from a Wikipedia 
# data collection of about 280,715 articles containing about 123,221,423 tokens. 
# What this code is trying to do with this parsed data is answer a collection of 
# questions/queries from the popular game show of Jeopardy in a text file called 
# question.txt. This Code aims to be able to use text retrieval strategies to answer 
# the jeopardy question correctly with prompts in the text file with a precision score
# of 40% or higher and a Mean Reciprocal Rank of 50% or higher. How this code achieved
# this goal was by creating different versions of the code, from a TF-IDF version 
# to the final version here, which is a BM25 ranking with a large language model 
# reranking prompting. This code first normalizes and parses the data from the Wikipedia
# collection while setting up the needed values to calculate a BM25 ranking in the 
# classes init. Then it takes this information in the main and sends it to the run_query,
# which does the same normalization as in the init to the query from the question.txt file 
# for every 100 queries of the Jeopardy game. After it normalizes the queries for each query,
# it sends them to _run_query, where the actual BM25 ranking of the queries is done. 
# In the BM25 ranking, it creates scores for each document based on its relevance to the query,
# and it returns the top 50 document titles with the highest relevance to the query back to the main. 
# Then it uses a large language model and prompts it to rerank the top 50 document titles based on 
# their relevance to the query, and rank the most relevant document title at index 0 and then this 
# reranked list is return to main where the precision and mean reciprocal rank of that query is 
# calculated and stored in the main. The code does this for each prompt/query in the question.txt file,
# and after all 100 are done, the final precision and mean reciprocal rank are shown to the user, 
# and the code finishes executing.


import openai
import collections
import math
import os
import re
import time

# you will need your own openai key to use the given LLM as used here
openai.api_key = "{your openai key}"

class IRSystem:
    def __init__(self, files):
       # these are BM25 set variable that control the saturation effect (k1)
       # and the influence of document length found using geekforgeek

        self.k1 = 1.2
        self.b = 0.75

        # docLens is a dictionary that stores the length of tokens 
        # of each document where the key is the document name and the value
        # is the number of tokens
        self.docLens = {}
        # avgDocLenth is a variable need to calculate the BM25 
        self.avgDocLength = 0
        # docWeights is a dictionary 
        self.docWeights = {}
        # df store the document frequencies which is the frequency of each token/term and the number
        # aka the number of document in which the token appears and it stores it as a dictionary
        # where the key is the term and the value is the number of doc it appears in
        self.df = collections.defaultdict(int)
        
        # total length is the number of tokens in the entire collection used to calculate avg Length
        totalLength = 0
        # finds the total number of documents in the entire collection used to calculate avgDocLenght
        docCount = 0

        # this loop parses and normiles the documents of the collection of wiki articles
        for filename in files:
            with open(os.path.join("./wiki-subset-20140602", filename), encoding="UTF-8") as f:
                docName = None
                tokens = []
                for line in f:
                    line = line.strip()
                    # the [[ ]] check here is how the document collection store and marks title/name of a doc
                    if line.startswith("[[") and line.endswith("]]"):
                        if docName is not None and tokens:
                            self._get_weights(docName, tokens)
                            totalLength += len(tokens)
                            docCount += 1
                        # start new article are found with the document titles,
                        # the start of a document title is denoted by "[[""]]" where
                        # the document name/title is in between this is captured and
                        # stored before tokenization without any processing and line[2:-2] skips the opening
                        # and closing braces when storing documents
                        docName = line[2:-2]
                        tokens = []

                    # add tokenized line to tokens and normalizes them
                    lowerLine = line.lower()
                    curTokens = re.sub(r'[^a-zA-Z0-9\s]', '', lowerLine).split()
                    for token in curTokens:
                        tokens.append(token)

                # process last article
                if docName is not None and tokens:
                    self._get_weights(docName, tokens)
                    totalLength += len(tokens)
                    docCount += 1
        # calculates the average document length for the BM25
        self.avgDocLength = totalLength / docCount

    def _get_weights(self, docName, tokens):
        # get frequencies of the tokens where the key is the token and value is its 
        # frequency of the token in the document this is the (tf)
        termFrequencies = collections.defaultdict(int)
        # this loop populates term frequency dictionary
        for token in tokens:
            # this if token is a validation check to make sure the token is not null or empty and if it isn't
            # add it to the term frequency
            if token:
                termFrequencies[token] += 1
        
        self.docLens[docName] = len(tokens)

        # loops through the terms/toksn in termFrequence
        for term in termFrequencies:
            # this makes sure the term is in the docWeights
            if term not in self.docWeights:
                self.docWeights[term] = collections.defaultdict(int)
            # stores the tf in the docWeight dictionary
            self.docWeights[term][docName] = termFrequencies[term]
            # increaes the document frequencies of the term by 1
            self.df[term] += 1

    def run_query(self, query):
       # this will normalized the query to match the same normalization of the
       # docs before making it a list of terms in the query and calling _run_query
       # on it where the BM25 is performed
        lowerQuery = query.lower()
        terms = re.sub(r'[^a-zA-Z0-9\s]', '', lowerQuery).split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # this will use a BM25 ranking to find similarity scores
        # to do this BM25 need to know N(num of doc in the directory)
        numOfDoc = len(self.docLens)
        # and the idf scores which is a dictionary here
        idfScores = collections.defaultdict(int)

        # this loop will calculate the idf for the terms in the query
        for term in terms:
            if self.df[term] != 0:
                # this does IDF for BM25 style where the goal is to divide the total numbe rof document in the directory
                # n(numOfDocs) - the number of document contian the term n_t(df of the term) +.5
                # and divide by n_t+.5  and takes the log of it this is differetnt for tf-idf since it only takes into conseration
                # of N/n_t and takes the log this is a more precise version of idf found equation at geekforGeeks
                idf = math.log10((numOfDoc - self.df[term] + 0.5) / (self.df[term] + 0.5))
                idfScores[term] = idf

        # this loop will get the similarity scores using the BM25 Ranking
        # where the key is the document name and the value is the BM25 score
        simScores = collections.defaultdict(int)
        for term in terms:
            if idfScores[term] > 0:
                for docName in self.docWeights[term]:
                    # this will get the term frequency from the docWeights
                    tf = self.docWeights[term][docName]
                    # this get the document Lenght used to make sure long document don't take over in the score
                    # BM25 ranking just because of thier sheer size
                    docLength = self.docLens[docName]
                    # this calculates the TF score where the numerator is the term frequency(tf) multiply by k1+1
                    # it does this to help correct/normilze the term frequecny 
                    # then the denominated is the tf+k1 agian to normilze and then the it multiply by (1 - self.b + self.b * (docLength / self.avgDocLength)
                    # which is a part to help prevent longer document form having an advantage in the ranking found equation at geeksForGeeks
                    tfScore = ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (docLength / self.avgDocLength))))
                    score = idfScores[term] * tfScore
                    # this will do the sum/do the sigma for the scores(idf*tf) which is the full BM25 score for a given term
                    simScores[docName] += score

        # sort and get top 50 sorts in reversed order so the higher more relevant scores are at the top of the ranking
        # [:50] cutoff the ranking to the top 50 results
        result = sorted(simScores, key=simScores.get, reverse=True)[:50]
        return result

    def calculate_mrr(self, results, correctAnswers):
        # this loop calculates the mean reciprocal rank 
        # what it does is use enumerate which is a python method
        # that will have a count starting from 0 (i in the loop)
        # with a variable attach to it in this case it the name of
        # a document  then it check the enumerated result to see 
        # if the answer is in the result and how far down the ranking
        # it is which is a good metric to see if your ranking is return
        # the most relevent or correct answer at the top of your ranking
        # which is why we use MRR and it will return 1/rank(rank =how far down correct result is)
        # if the mrr is found otherwise it will return 0
        rank = -1
        for i, answer in enumerate(results):
            if answer in correctAnswers:
                rank = i + 1
                return 1/rank
        return 0

  
def rerank_with_llm(query, doc_titles, batch_size=50, max_retries=3):
    reranked_scores = []
    for i in range(0, len(doc_titles), batch_size):
        batch = doc_titles[i:i+batch_size]
        prompt = (
            f"You are a trivia expert. Given the Jeopardy question:\n\n"
            f"\"{query}\"\n\n"
            f"Which of the following Wikipedia article titles is the best possible answer? "
            f"Rerank all 50 titles in order of which is most likely the answer, with the most likely first. Give only the article names as they are written, no numbers.\n"
        )
        for j, title in enumerate(batch):
            prompt += f"{j+1}. {title}\n"

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                break  # success, break out of retry loop
            except openai.error.OpenAIError as e:
                print(f"OpenAI error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        else:
            continue  # all retries failed, skip batch

        answers = response['choices'][0]['message']['content'].split("\n")
        for answer in answers:
            reranked_scores.append(answer[3:].strip())

        time.sleep(1)  # avoid hitting rate limits

    return reranked_scores if reranked_scores else doc_titles


def main():
    ir = IRSystem(os.listdir("./wiki-subset-20140602"))
    # counter count the number of correct document titles found at index 0 for precision in the question.txt file
    counter = 0
    # mrrSum totals the MRR of each query to find the MRR of all query in the question.txt file
    mrrSum = 0

    with open('questions.txt', 'r') as f:
        lines = f.readlines()
        # this loop through the lines in the question.txt file 4 lines at a time
        # it does this because the question are formatted with the category with a /n at top
        # followed by the question/query with a /n after and finally the answer/answer title
        # used to check Precision after and a then a blank /n by looping over 4 at a time
        # it allows us to isolate the question and parse them as need and call the need function
        # based on the queries
        for i in range(0, len(lines), 4):
            if i + 2 >= len(lines):
                break
            queryLine = [lines[i].strip().lower(), lines[i+1].strip().lower()]
            query = re.sub(r'[^a-zA-Z0-9\s]', '', queryLine[0] +' '+queryLine[1])
            results = ir.run_query(query)

            rerankedResults = rerank_with_llm(query, results)

            # check against answers in question to see if the answer
            # is at the top ranked doc
            answerLine = lines[i+2].strip()
            # splits on | since | marks that there is more than one
            # acceptable answer
            if '|' in answerLine:
                answers = answerLine.split('|')
            else:
                answers = [answerLine]
            # this loop goes through and see if the rerank list at index 0
            # aka the top most relevent doc name is in answer from the txt file
            # answer if so it add to correct answer and add to the total count of 
            # correct answers 
            countCorrectAnswer = 0
            for answer in answers:
                if answer == rerankedResults[0]:
                    countCorrectAnswer += 1
            counter += countCorrectAnswer
            mrrSum += ir.calculate_mrr(rerankedResults, answers)

    # this prints the final precision where counter is the number of correct document found after reranking the 
    # results at index 0 from the llm / 100 which is the number of question/query asked
    print(f'Precision: {counter / 100}')
    # this prints the final MRR where mrrSum is the running total of MRR from each query  
    # / 100 which is the number of question/query asked
    print(f'Mean Reciprocal Rank (MRR): {mrrSum / (100)}')

if __name__ == '__main__':
    main()
