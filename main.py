#Evaluate multilingual pretrained model using UCCA annoations for semantic role probing
from lxml import etree
import os
from transformers import BertTokenizerFast, BertModel
import torch
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.metrics import accuracy_score

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


def get_sentence_from_xml(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Extract terminal nodes
    terminal_nodes = {}
    words = []
    for node in root.xpath('.//layer[@layerID="0"]//node[@type="Word" or @type="Punctuation"]'):
        node_id = node.attrib["ID"]
        text = node.find("attributes").attrib.get("text", "")
        terminal_nodes[node_id] = text
        words.append(text)
    
    new_sentence = decode_text(" ".join(words))

    return new_sentence


def get_sentence_embeddings(sentence, corresponding_word_roles, tokenizer, model, layer):
    if isinstance(sentence, str):
        words = sentence.split()
    else:
        words = sentence

    # Tokenize
    tokens = tokenizer(words, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of all layer outputs

    hidden_states = hidden_states[layer] # Specify the layer of the model that we want to get the embeddings from

    # Align subwords to words and aggregate embeddings
    word_embeddings = []
    word_ids = tokens.word_ids()  # Maps subwords to word indices
    current_word = None
    current_embeddings = []

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if current_word is None:
            current_word = word_id

        if word_id != current_word:
            # Aggregate subword embeddings for the current word
            word_embedding = torch.mean(torch.stack(current_embeddings), dim=0)
            word_embeddings.append(word_embedding)
            current_embeddings = []
            current_word = word_id

        # Collect subword embeddings
        current_embeddings.append(hidden_states[0, i])

    # Handle the last word
    if current_embeddings:
        word_embedding = torch.mean(torch.stack(current_embeddings), dim=0)
        word_embeddings.append(word_embedding)

    # Validate word_embeddings length, if the word length and embeddings are not the same, don't include it in the output
    if len(word_embeddings) != len(words):
        #print(words)
        return []

    # Map embeddings to words and roles
    result = []
    for word_idx, (word, roles) in enumerate(zip(words, corresponding_word_roles)):
        for role in roles:  # If a word has multiple roles
            result.append({
                "word_idx": word_idx,
                "token": word,
                "role": role,
                "embedding": word_embeddings[word_idx].numpy()
            })

    return result


def flatten_data(data):
    '''
    Flatten a list of list of dictionaries into a list
    '''
    X = []
    y = []
    for sentence in data:
        for word_data in sentence:
            X.append(word_data["embedding"])
            y.append(word_data["role"])
    return np.array(X), y


def map_roles_to_terminals(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Map each node ID to its terminal nodes
    node_to_terminals = {}
    for node in root.xpath('.//layer[@layerID="1"]//node'):
        node_id = node.attrib.get("ID")
        terminal_edges = node.xpath('./edge[@type="Terminal"]')  # Direct children only
        node_to_terminals[node_id] = [
            edge.attrib.get("toID") for edge in terminal_edges
        ]

    # Map terminal node IDs to words
    node_id_to_word = {}
    for node_id, terminal_ids in node_to_terminals.items():
        terminals = []
        for terminal_id in terminal_ids:
            terminal_node = root.xpath(
                f'.//layer[@layerID="0"]//node[@ID="{terminal_id}"]'
            )
            if terminal_node:
                terminal_node = terminal_node[0]
                text = terminal_node.find("attributes").attrib.get("text", "")
                position = terminal_node.find("attributes").attrib.get(
                    "paragraph_position", ""
                )
                terminals.append((text, position))
        node_id_to_word[node_id] = terminals

    # Map roles to words
    node_type_to_word = {}
    for node_id, words in node_id_to_word.items():
        edges = root.xpath(
            f'.//layer[@layerID="1"]//node//edge[@toID="{node_id}"]'
        ) 
        for edge in edges:
            edge_type = edge.attrib.get("type")
            if edge_type not in node_type_to_word:
                node_type_to_word[edge_type] = []
            node_type_to_word[edge_type].extend(words)

    for role in node_type_to_word:
        node_type_to_word[role] = list(set(node_type_to_word[role]))

    return node_type_to_word


def create_ordered_list(data, sentence):
    '''
    Create a dictionary which the keys are the words of the sentence IN ORDER and the values are the UCCA role(s) associated with each word
    '''
    word_roles = {}
    split_sentence = sentence.split()
    for i, word in enumerate(split_sentence):
        roles = []
        word_tuple = (word, str(i+1))
        for role in data:
            if word_tuple in data[role]:
                roles.append(role)
        word_roles[word_tuple] = roles
    return word_roles


def analyze_role_accuracy(y_true, y_pred):
    '''
    Returns the accuracy spit by the specific UCCA roles 
    
    NOT CURRENTLY PRINTED, BUT IF WE WANT IN-DEPTH ANALYSIS, WE CAN USE THIS
    '''
    role_results = defaultdict(list)
    for true, pred in zip(y_true, y_pred):
        role_results[true].append(1 if true == pred else 0)

    for role, results in role_results.items():
        accuracy = sum(results) / len(results)
        print(f"Role: {role}, Accuracy: {accuracy:.2f}")


def analyze_embedding_layers(layer, english_data, french_data):
    '''
    Run the analysis on each layer of mBERT (12 total)
    '''
    print("LAYER:", layer)
    print("\n")
    # Get mBERT word level embeddings
    english_sentence_embeddings = []
    french_sentence_embeddings = []
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    for data in english_data:
        sentence = []
        roles_per_word = []
        for word, position in data:
            sentence.append(word)
            roles_per_word.append(data[(word, position)])
        english_sentence_embeddings.append(get_sentence_embeddings(sentence, roles_per_word, tokenizer, model, layer))
    
    for data in french_data:
        sentence = []
        roles_per_word = []
        for word, position in data:
            sentence.append(word)
            roles_per_word.append(data[(word, position)])
        french_sentence_embeddings.append(get_sentence_embeddings(sentence, roles_per_word, tokenizer, model, layer))

    # Flatten the lists into emeddings mapped to the UCCA roles
    X_english, y_english = flatten_data(english_sentence_embeddings)
    X_french, y_french = flatten_data(french_sentence_embeddings)

    # Train and test on English data
    # Split English data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_english, y_english, test_size=0.2, random_state=42)

    # Use a Logistic Regression model
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial", class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate on English test data
    y_pred = clf.predict(X_test)
    print("Evaluation on English Test Data:")
    print(accuracy_score(y_test, y_pred))

    y_pred_french = clf.predict(X_french)
    print("Cross-Lingual Evaluation (Train on English, Test on French):")
    print(accuracy_score(y_french, y_pred_french))
    '''
    y_pred_english = clf.predict(X_english)
    print("Role Specific Accuracy English")
    analyze_role_accuracy(y_english, y_pred_english)
    '''

    # Split French data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_french, y_french, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train, y_train)

    # Evaluate on English test data
    y_pred = clf.predict(X_test)
    print("Evaluation on French Test Data:")
    print(accuracy_score(y_test, y_pred))

    y_pred_english = clf.predict(X_english)
    print("Cross-Lingual Evaluation (Train on French, Test on English):")
    print(accuracy_score(y_english, y_pred_english))
    '''
    y_pred_french = clf.predict(X_french)
    print("Role Specific Accuracy French")
    analyze_role_accuracy(y_french, y_pred_french)
    '''


def main():
  #Map UCCA roles to word in sentences
  file_path = "potential_corpus/corpus/passage"
  file_end = ".xml"
  ordered_words_and_roles_en = []
  for i in range(36, 63):
      fp = file_path + str(i) + file_end
      ordered_words_and_roles_en.append(create_ordered_list(map_roles_to_terminals(fp), get_sentence_from_xml(fp)))
  for i in range(286, 319):
      fp = file_path + str(i) + file_end
      ordered_words_and_roles_en.append(create_ordered_list(map_roles_to_terminals(fp), get_sentence_from_xml(fp)))
  for i in range(814, 847):
      fp = file_path + str(i) + file_end
      ordered_words_and_roles_en.append(create_ordered_list(map_roles_to_terminals(fp), get_sentence_from_xml(fp)))
  for i in range(880, 910):
      fp = file_path + str(i) + file_end
      ordered_words_and_roles_en.append(create_ordered_list(map_roles_to_terminals(fp), get_sentence_from_xml(fp)))
  for i in range(968, 999):
      fp = file_path + str(i) + file_end
      ordered_words_and_roles_en.append(create_ordered_list(map_roles_to_terminals(fp), get_sentence_from_xml(fp)))

  file_path = "potential_corpus/corpus/passage"
  file_end = ".xml"
  ordered_words_and_roles_fr = []
  for i in range(77, 104):
      ordered_words_and_roles_fr.append(create_ordered_list(map_roles_to_terminals(file_path + str(i) + file_end), get_sentence_from_xml(file_path + str(i) + file_end)))
  for i in range(416, 449):
      ordered_words_and_roles_fr.append(create_ordered_list(map_roles_to_terminals(file_path + str(i) + file_end), get_sentence_from_xml(file_path + str(i) + file_end)))
  for i in range(764, 797):
      ordered_words_and_roles_fr.append(create_ordered_list(map_roles_to_terminals(file_path + str(i) + file_end), get_sentence_from_xml(file_path + str(i) + file_end)))
  for i in range(848, 878):
      ordered_words_and_roles_fr.append(create_ordered_list(map_roles_to_terminals(file_path + str(i) + file_end), get_sentence_from_xml(file_path + str(i) + file_end)))
  for i in range(911, 942):
      ordered_words_and_roles_fr.append(create_ordered_list(map_roles_to_terminals(file_path + str(i) + file_end), get_sentence_from_xml(file_path + str(i) + file_end)))

  # Get mBERT word level embeddings for each layer
  for i in range(1, 13):
      analyze_embedding_layers(i, ordered_words_and_roles_en, ordered_words_and_roles_fr)


main()

                                               
                                    