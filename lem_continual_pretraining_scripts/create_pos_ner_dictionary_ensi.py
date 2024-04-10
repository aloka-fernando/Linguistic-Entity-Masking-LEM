import json
import pickle
import time
import torch
import re

import numpy as np
from io import BytesIO
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


#debug
debug=False

#clear cache
torch.cuda.empty_cache()
print("Cleared cuda cache.")

#time
startTime=time.time()
#v2 - ner only
#v3 - ner, new_wo_nums

#parameters
data_dir="/userdirs/aloka/p31_pretrained_models_ft/sentence-transformers/examples/unsupervised_learning/MLM/data/gov"
wrk_dir="/userdirs/aloka/p31_pretrained_models_ft/4_ner_pos_exp/data"
input_file_names = ["en_si_train_tlm.txt", "en_si_valid_tlm.txt"]
data_out_subdir="data_out"
out_file_name="ne_pos-tlm-dict_v5.en-si.pkl"
src="en"
tgt="si"

# initialize roberta tokenizer
model_name="xlm-roberta-base"
xlmr_tokenizer=AutoTokenizer.from_pretrained(model_name)

#si words pos tags
si_noun_tags = ["NNC", "NNP"]
si_verb_tags = ["VNN", "VFM", "VNF", "VP"]
si_pos_dictionary_file="/userdirs/aloka/p31_pretrained_models_ft/4_ner_pos_exp/pos_dicts/train100KDD-tu.si.pos-all-words.dict"
si_pos_dict={}
with open(si_pos_dictionary_file, 'rb') as pos_file:
    si_pos_dict = pickle.load(pos_file)
print('Total entries in the Si ne_pos_dict : {}'.format(len(si_pos_dict)))


#en pos_tag_dict path
en_verb_pos_tags=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
en_noun_pos_tags=['NN', 'NNP', 'NPPS', 'NNP', 'NNPS', 'NNS']
en_pos_dict_path="/userdirs/aloka/p31_pretrained_models_ft/4_ner_pos_exp/pos_dicts/train100KDD-tu.pos_only.en.pkl"
en_pos_dict={}
with open(en_pos_dict_path, 'rb') as pos_file:
    en_pos_dict = pickle.load(pos_file)
print('Total entries in the En pos_dict : {}'.format(len(en_pos_dict)))


#check digit
def is_number(token):
    pattern = r'^_?▁?,?\(?[-+$]?[0-9]*\.?\⁄?[0-9,]+%?\)?'
    return bool(re.match(pattern, token))


#ner model
ner_model_path="/userdirs/aloka/p31_pretrained_models_ft/4_ner_pos_exp/ner_models/xlm-roberta-base-conll-finetuned-finetuned-sinhala-english-tamil"
tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
model = AutoModelForTokenClassification.from_pretrained(ner_model_path, num_labels=len(label_list))
trainer = Trainer(model=model)
trainer.model = model.cuda()

#returns the sentence by merging the tokens
def get_seq_for_dict_key(sentence):

    sentence_inputs=xlmr_tokenizer.tokenize(sentence, max_length=100, truncation=True ) #add_special_tokens=True to add <s> and </s>
    sentence_words=[]

    for token in sentence_inputs:
        if token == "<s>" or token == "</s>":
            continue
        if token.startswith("▁"):
            # print('[_whole_word_mask] DebugCheckpoint#3 ELSE > if len(cand_indexes) >= 1 and token.startswith("##") : {}\n'.format("False"))
            sentence_words.append(token[1:])
        else:
            # print('[_whole_word_mask] DebugCheckpoint#5 if len(cand_indexes) >= 1 and not token.startswith("_") : {}'.format("True"))
            sentence_words[-1] = sentence_words[-1] + token
    sentence_text = ' '.join(sentence_words)

    return sentence_text, sentence_words

#returns the NE, NE-span and NE-type lists
def ne_recognizer(splitting):

    try:
        # NER
        tokenized_inputs = tokenizer(splitting, truncation=True, is_split_into_words=True, max_length=512)

        word_ids = tokenized_inputs.word_ids()
        prediction, label, _ = trainer.predict([tokenized_inputs])
        prediction = np.argmax(prediction, axis=2)
        true_prediction_long = [label_list[p] for p in prediction[0]]
        true_prediction = []

        ner_prediction = []  # the entire list with ner tags
        ner_indexes = []
        ner_tokens = []
        for y in range(0, len(splitting)):
            true_prediction.append(true_prediction_long[word_ids.index(y)])
            if true_prediction_long[word_ids.index(y)] != 'O' and not is_number(splitting[y]):
                ner_indexes.append(y)
                ner_tokens.append(splitting[y])
        #print('NER of sequence : {}'.format(true_prediction))
        return true_prediction, ner_indexes, ner_tokens

    except:
        # print(tokenized_inputs)
        # print(word_ids)
        # print(true_prediction_long)
        return true_prediction, ner_indexes, ner_tokens


#pos_tagging the sentence
def pos_tag_sequence(sentence_words):

    pos_tags_with_tokens=[]
    pos_tags=[]
    verb_indexes = []
    noun_indexes = []
    for i in range(0, len(sentence_words)):

        if sentence_words[i] in en_pos_dict:
            pos_tags_with_tokens.append('[{}]{}/{}'.format(i, sentence_words[i], en_pos_dict[sentence_words[i]]))
            pos_tags.append(en_pos_dict[sentence_words[i]])

            if en_pos_dict[sentence_words[i]] in en_verb_pos_tags:
                verb_indexes.append(i)
            elif en_pos_dict[sentence_words[i]] in en_noun_pos_tags:
                noun_indexes.append(i)

        elif sentence_words[i] in si_pos_dict:
            pos_tags_with_tokens.append('[{}]{}/{}'.format(i, sentence_words[i], si_pos_dict[sentence_words[i]]))
            pos_tags.append(si_pos_dict[sentence_words[i]])

            if si_pos_dict[sentence_words[i]] in si_verb_tags:
                verb_indexes.append(i)
            elif si_pos_dict[sentence_words[i]] in si_noun_tags:
                noun_indexes.append(i)

        else:
            pos_tags_with_tokens.append('[{}]{}/{}'.format(i, sentence_words[i], "NONE"))
            pos_tags.append("NONE")

    return pos_tags_with_tokens, pos_tags, verb_indexes, noun_indexes



#create dict
ner_pos_dict={} #{sent: {splitting : [], }}

#main function

for input_file_name in input_file_names:
    for index, sentence_text in enumerate(open("{}/{}".format(wrk_dir, input_file_name), "r", encoding="utf8")):

           print('\n[{}]'.format(index))
           sentence_text=sentence_text.strip()
           sentence_text_for_dict, sentence_words = get_seq_for_dict_key(sentence_text)

           splitting = sentence_text.split()
           ner_pos_dict[sentence_text_for_dict] = {}
           ner_pos_dict[sentence_text_for_dict]["splitting"]=splitting
           ner_pos_dict[sentence_text_for_dict]["sentence_words_from_tok"] = sentence_words

           #ner
           true_prediction, ner_indexes, ner_tokens = ne_recognizer(sentence_words)
           ner_pos_dict[sentence_text_for_dict]["ner_prediction_with_tokens"] = ['[{}]{}'.format(i, sentence_words[i]) for i, token in enumerate(true_prediction)]
           ner_pos_dict[sentence_text_for_dict]["ner_prediction"] = true_prediction
           ner_pos_dict[sentence_text_for_dict]["ner_tokens"] = ['[{}]{}'.format(i, sentence_words[i]) for i in ner_indexes]
           ner_pos_dict[sentence_text_for_dict]["ner_indexes"] = ner_indexes

           #pos
           pos_tags_with_tokens=[]
           pos_tags=[]
           verb_indexes=[]
           noun_indexes=[]

           pos_tags_with_tokens, pos_tags, verb_indexes, noun_indexes = pos_tag_sequence(sentence_words)

           ner_pos_dict[sentence_text_for_dict]["pos_tags"] = pos_tags
           ner_pos_dict[sentence_text_for_dict]["pos_tags_with_tokens"] = pos_tags_with_tokens
           ner_pos_dict[sentence_text_for_dict]["verb_indexes"] = verb_indexes
           ner_pos_dict[sentence_text_for_dict]["verb_tokens"] = ['[{}]{}'.format(i, sentence_words[i]) for i in verb_indexes]
           ner_pos_dict[sentence_text_for_dict]["noun_indexes"] = noun_indexes
           ner_pos_dict[sentence_text_for_dict]["noun_tokens"] = ['[{}]{}'.format(i, sentence_words[i]) for i in noun_indexes]

           if debug:
               print('[SENT_TXT] {}'.format(sentence_text_for_dict)) if debug else None
               print('[sentence_text_for_dict] {}'.format(sentence_text_for_dict)) if debug else None
               print('[sentence_words] {}'.format(sentence_words)) if debug else None
               print('[true_prediction] {}'.format(true_prediction)) if debug else None
               print('[ner_indexes] {}'.format(ner_indexes)) if debug else None
               print('[ner_tokens] {}'.format(ner_tokens)) if debug else None
               print('[pos_tags_with_tokens] {}'.format(pos_tags_with_tokens)) if debug else None
               print('[pos_tags] {}'.format(pos_tags)) if debug else None
               print('[verb_indexes] {}'.format(verb_indexes)) if debug else None
               print('[noun_indexes] {}'.format(noun_indexes)) if debug else None
               print('[pos_tags==sentence_words] {}'.format(len(pos_tags)==len(sentence_words))) if debug else None
               print('----------------------------------------------------------------------------------------')



print('\n\nNumber of entries in the ner_pos_dict : {}'.format(len(ner_pos_dict)))
#print('\nNER_POS_DICT\n{}'.format(ner_pos_dict))

#write dictionary into a json file-verification

#write dictionary into a json file-verification
with open("{}/{}/text_files/{}.{}.txt".format(wrk_dir, data_out_subdir, out_file_name, src), "w") as outfile_csv:
    for key in ner_pos_dict.keys():
        outfile_csv.write('{}: {}\n'.format(key, ner_pos_dict[key]))

# write
with open("{}/{}/dict_files/{}.{}.pkl".format(wrk_dir, data_out_subdir, out_file_name, src), "wb") as outfile:
    pickle.dump(ner_pos_dict, outfile)

elapsedTime=time.time()-startTime

hours=int(elapsedTime//3600)
mints=int((elapsedTime%3600)//60)
secs=int((elapsedTime%3600)%60)
print('Elapsed time : {}hrs {}min {}sec'.format(hours, mints, secs))

print('Script completed.......')

#Time taken
#En-valid
# Elapsed time : 0hrs 1min 30sec
#En-train
#Elapsed time : 0hrs 30min 41sec

