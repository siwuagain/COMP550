from utils import logger
import preprocessor as pp
from transformers import AutoTokenizer, AutoModel
import numpy as np
import stanza
import networkx as nx
import h5py
import nltk
from nltk.tokenize import sent_tokenize
import torch
import re
from lxml import etree
import pickle
from nltk.tree import Tree

pp = pp.PreProcessor()
logger = logger.Logger()


def extract_sentence_from_xml(file_path):
    '''
    Getting the sentences from the xml files were the best move because sentences from the text files don't exactly correlate to each other (due to different spacing, punctuation, etc)
    '''
    tree = etree.parse(file_path)
    root = tree.getroot()
    words = []
    nodes = root.xpath('.//node[@type="Word" or @type="Punctuation"]')
    for node in nodes:
        attributes = node.find("attributes")
        if attributes is not None:
            text = attributes.get("text")
            if text:
                words.append(text)
    sentence = " ".join(words)
    sentence = sentence.replace("...", "")
    sentence = sentence.replace(". . .", "")
    return sentence


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


def load_english_sentences():
    '''
    loads all the sentences from the english passages
    '''
    sentences = []
    file_path = "potential_corpus/corpus/passage"
    file_end = ".xml"
    for i in range(36, 63):
        fp = file_path + str(i) + file_end
        sentences.append(extract_sentence_from_xml(fp))
    for i in range(286, 319):
        fp = file_path + str(i) + file_end
        sentences.append(extract_sentence_from_xml(fp))
    for i in range(814, 847):
        fp = file_path + str(i) + file_end
        sentences.append(extract_sentence_from_xml(fp))
    for i in range(880, 910):
        fp = file_path + str(i) + file_end
        sentences.append(extract_sentence_from_xml(fp))
    for i in range(968, 999):
        fp = file_path + str(i) + file_end
        sentences.append(extract_sentence_from_xml(fp))
    return sentences


def load_french_sentences():
    '''
    loads all the sentences from the french passages
    '''
    sentences = []
    file_path = "potential_corpus/corpus/passage"
    file_end = ".xml"
    for i in range(77, 104):
        sentences.append(extract_sentence_from_xml(file_path + str(i) + file_end))
    for i in range(416, 449):
        sentences.append(extract_sentence_from_xml(file_path + str(i) + file_end))
    for i in range(764, 797):
        sentences.append(extract_sentence_from_xml(file_path + str(i) + file_end))
    for i in range(848, 878):
        sentences.append(extract_sentence_from_xml(file_path + str(i) + file_end))
    for i in range(911, 942):
        sentences.append(extract_sentence_from_xml(file_path + str(i) + file_end))
    return sentences


def decode_text(sentence):
    '''
    The text written in the xml is encoded so this function changes its code into actual english/french characters
    '''
    unicode_changes = {
    "2019": "’",  # Right single quotation mark
    "00e9": "é",  # Latin small letter e with acute
    "00e0": "à",  # Latin small letter a with grave
    "00e8": "è",  # Latin small letter e with grave
    "00f4": "ô",  # Latin small letter o with circumflex
    "00c7": "Ç",  # Latin capital letter C with cedilla
    "00e7": "ç",  # Latin small letter c with cedilla
    "00f9": "ù",  # Latin small letter u with grave
    }
    unicode_pattern = r"\\u([0-9a-fA-F]{4})"
    
    def replace_match(match):
        code = match.group(1)
        return unicode_changes.get(code, match.group(0))
    
    return re.sub(unicode_pattern, replace_match, sentence)


def get_embeddings(sentences, tokenizer, model):
    '''
    Gets the embeddings of the sentences
    '''
    embeddings = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, return_tensors="pt")
        output = model(**encoded_input)
        embeddings.append(output)
    return embeddings


def get_parse_tree_stanza(sentences, enlgish):
    '''
    Uses the Stanza library to get the parse tree and its corresponding matrix (not from stanza, computed by hand so kinda iffy) for each sentence (english and french)
    '''
    trees = []
    matrices = []

    nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse')
    if enlgish:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        
    for sentence in sentences:
            doc = nlp(sentence)
            trees.append(doc)

            for sent in doc.sentences:
                num_tokens = len(sent.words)
                graph = nx.Graph()
                for word in sent.words:
                    if word.head > 0:
                        graph.add_edge(word.id - 1, word.head - 1)

                distance_matrix = np.zeros((num_tokens, num_tokens), dtype=float)
                shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
                for i in range(num_tokens):
                    for j in range(num_tokens):
                        if i in shortest_paths and j in shortest_paths:
                            distance_matrix[i, j] = shortest_paths[i][j]
                        else:
                            #this typically happens for PUNT word types
                            distance_matrix[i, j] = float("inf")

                matrices.append(distance_matrix)
    return trees, matrices


def save_embeddings_to_hdf5(embeddings, output_file):
    '''
    Not really used, but just saves all the embeddings into a file
    '''
    with h5py.File(output_file, "w") as f:
        for idx, embedding in enumerate(embeddings):
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            f.create_dataset(str(idx), data=embedding)


def visualize_stanza(tokens, root_id):
    root = next(token for token in tokens if token['id'] == root_id)
    children = [token for token in tokens if token['head_id'] == root_id]
    return Tree(f"{root['text']} ({root['deprel']})", [visualize_stanza(tokens, child['id']) for child in children])


def main():
    #Import sentences (and make sure each list element is its own sentence)
    '''
    All this is literally just to ensure each sentence within each list index maps from english to french
    '''
    english_sentences = load_english_sentences()
    french_sentences = load_french_sentences()
    parallel_english_sentences = []
    parallel_french_sentences = []
    for i in range(len(english_sentences)):
        punctuations = {'.', '!', '?'}
        if sum(1 for char in english_sentences[i] if char in punctuations) == sum(1 for char in french_sentences[i] if char in punctuations):
            parallel_english_sentences.append(english_sentences[i])
            parallel_french_sentences.append(french_sentences[i])
    
    nltk.download('punkt')
    split_en_sentences = []
    split_fr_sentences = []
    for j in range(len(parallel_english_sentences)):
        en_sentences = sent_tokenize(parallel_english_sentences[j])
        fr_sentences = sent_tokenize(parallel_french_sentences[j])
        if len(en_sentences) != len(fr_sentences):
            if len(en_sentences)-1 == len(fr_sentences):
                en_sentences.pop()
            elif len(fr_sentences)-1 == len(en_sentences):
                fr_sentences.pop()
        
        split_en_sentences.extend(en_sentences)
        split_fr_sentences.extend(fr_sentences)
    
    cleaned_en_sentences = []
    cleaned_fr_sentences = []
    for k in range(len(split_en_sentences)):
        new_text = decode_text(split_en_sentences[k])
        cleaned_en_sentences.append(new_text)
        new_text = decode_text(split_fr_sentences[k])
        cleaned_fr_sentences.append(new_text)
    
    print(len(cleaned_en_sentences))
    print(len(cleaned_fr_sentences))


    #Sentence preprocessing
    '''
    Do more preprocessing or naw?
    '''

    #Generate parse trees for the sentences
    stanza.download("en")
    stanza.download("fr")
    en_sentence_parse_trees, en_sentence_matrix = get_parse_tree_stanza(cleaned_en_sentences, True)
    fr_sentence_parse_trees, fr_sentence_matrix = get_parse_tree_stanza(cleaned_fr_sentences, False)
    #tree = visualize_stanza(en_sentence_parse_trees[0], 5)
    #tree.pretty_print()
    
    #print(print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in en_sentence_parse_trees[0].sentences for word in sent.words], sep='\n'))
    #print(print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in fr_sentence_parse_trees[0].sentences for word in sent.words], sep='\n'))
    '''
    with open("sentence_data.pkl", "wb") as f:
        pickle.dump({
            "en_sentence_parse_trees": en_sentence_parse_trees,
            "en_sentence_matrix": en_sentence_matrix,
            "fr_sentence_parse_trees": fr_sentence_parse_trees,
            "fr_sentence_matrix": fr_sentence_matrix
        }, f)
    '''
    #print(en_sentence_matrix[0])
    #print(fr_sentence_matrix[0])
    print(len(en_sentence_parse_trees))
    print(len(fr_sentence_parse_trees))


    #Embed sentences
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    english_embeddings = get_embeddings(cleaned_en_sentences, tokenizer, model)
    french_embeddings = get_embeddings(cleaned_fr_sentences, tokenizer, model)
    print(len(english_embeddings))
    print(len(french_embeddings))
    '''
    save_embeddings_to_hdf5(english_embeddings, "english_embeddings.h5")
    save_embeddings_to_hdf5(french_embeddings, "french_embeddings.h5")
    '''

    #Split embeddings into training data, development data, and test data

    #Train structural probe
    '''
    Find a way to compare the probe's output to the Stanza parse trees
    '''

    #Evaluate on test data


main()


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