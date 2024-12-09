from utils import logger
import preprocessor as pp

pp = pp.PreProcessor()
logger = logger.Logger()

def english_preprocess(sentence):
    '''
    Lowercasing
    Remove punctuation (not needed for mBERT but might need for XLM-R)
    Tokenization
    '''
    pass

def french_preprocess(sentence):
    '''
    Lowercasing
    Removing accents
    Remove punctuation (not needed for mBERT but might need for XLM-R)
    Tokenization
    '''
    pass




'''
USING THE MODEL
1. Tokenize preprocessed sentences
2. Extract embeddings using mBERT or XLM-R

USING THE PROBE
The probe needs embeddings as input, and corresponding syntactic properties as output/labels
0. Split the embeddings into training, development, and test set?
1. Get syntactic annotations like parse tree distances or depths, derived from the corpus.
        (this might be in the xml file of the corpus, or we have to do this manually. how? idk.)
2. Train the structural probe (the one linked in final project proposal)
        (modify the parameters of the probe to see which one has the best result)
3. Evaluate the probe on the test set
        (use spearman correlation, accuracy, etc)
4. Run the probe on both english and french embeddings

EVALUATING RESULTS
Syntactic Differences:
1. Analyze how well embeddings from each language encode syntactic properties.
Differences in Spearman correlation can reveal cross-linguistic syntactic variations.
'''