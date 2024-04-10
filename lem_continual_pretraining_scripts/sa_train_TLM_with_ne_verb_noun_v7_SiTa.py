"""
This file runs Masked Language Model. You provide a training file. Each line is interpreted as a sentence / paragraph.
Optionally, you can also provide a dev file.

The fine-tuned model is stored in the output/model_name folder.

Usage:

sentence-transformers

python train_mlm.py model_name data/train_sentences.txt [data/dev_sentences.txt]
"""
import pickle
import re
from random import randint
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import numpy as np
import random
import time

#time
startTime=time.time()

torch.cuda.empty_cache()
print("Cleared cuda cache.")


#import logging
import sys
import gzip
from datetime import datetime


#debuging switches
torch_call_debug_mode=False
whole_word_masking_debug_mode=False
torch_collate_batch_debug_mode=False
torch_mask_tokens_debug_mode=False
ner_recognizer_debug_mode=False
print_final_masking_inputs=False

#inputs
data_path='/path/to/data'
train_file_name = "src/sentences.txt"
valid_file_name = "tgt/sentences.txt"
ne_pos_dict_file ="/path/to/ne-pos-noun/dict/file.pkl"
ne_pos_dict={}

#parameters
#output and training parameters
output_dir = "/output/directory"
num_train_epochs=55
per_device_train_batch_size=32
save_steps=5000
max_length=100
use_fp16=True
global_mlm = True
global_mlm_prob = 0.15
ne_mlm_prob = 1.0
verb_mlm_prob = 1.0
noun_mlm_prob = 0.0

# Load the xlm-roberta model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#ner model
ner_model_path="file/path/to/ner/multilingual/model"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path, num_labels=len(label_list))
ner_trainer = Trainer(model=ner_model)
ner_trainer.model = ner_model.cuda()


#xlm-roberta tokenizer
# Print all special tokens
print(tokenizer.all_special_tokens)
# Print all special token ids
print(tokenizer.all_special_ids)
print('\n\n')

#ner,pos dictionary
with open(ne_pos_dict_file, 'rb') as ner_pos_file:
    ne_pos_dict = pickle.load(ner_pos_file)
print('Total entries in the ne_pos_dict : {}'.format(len(ne_pos_dict)))

#output_dir = "experiments/{}-{}".format(model_name.replace("/", "_"),  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)

#read train-file and print no of lines
train_sentences =[line.strip() for line in open("{}/{}".format(data_path, train_file_name), "r", encoding="utf8")]
print('No of train file lines : {}'.format(len(train_sentences)))

#read valid-file and print no of lines
valid_sentences =[line.strip() for line in open("{}/{}".format(data_path, valid_file_name), "r", encoding="utf8")]
print('No of valid_sentences lines : {}'.format(len(valid_sentences)))

#add index to masked lm, covered indexes
def add_subword_from_item_list_indexes_to_masked_lms(item_list_index, covered_indexes, masked_lms):
    is_any_index_covered=False

    for index in item_list_index:
        if index in covered_indexes:
            is_any_index_covered = True
            return covered_indexes, masked_lms

    if is_any_index_covered:
        return covered_indexes, masked_lms

    random_list_item_index = random.randint(0, len(item_list_index)-1)
    #print('item_list_index {}, index {}'.format(item_list_index, random_list_item_index))
    subword_index_from_item_list_index =item_list_index[random_list_item_index]
    covered_indexes.update([index for index in item_list_index])
    masked_lms.append(subword_index_from_item_list_index)
    return covered_indexes, masked_lms


#check digit
def is_number(token):
    pattern = r'^_?▁?,?\(?[-+$]?[0-9]*\.?\⁄?[0-9,]+%?\)?'
    return bool(re.match(pattern, token))

#pos_tagging the sentence
def ta_pos_tag_sequence(sentence_words):
    verb_indexes = []
    noun_indexes = []
    for i in range(0, len(sentence_words)):

        if sentence_words[i] in ne_pos_dict:
            if ne_pos_dict[sentence_words[i]]=="VERB":
                verb_indexes.append(i)
            elif ne_pos_dict[sentence_words[i]]=="NOUN":
                noun_indexes.append(i)
    return verb_indexes, noun_indexes

# ner function to call if the sequence is not available as a key in the dictionary
def ne_recognizer(splitting, ner_tokenizer, ner_trainer):
    # NER
    print('Spliltting : {}'.format(splitting)) if ner_recognizer_debug_mode else None
    ner_tokenized_inputs = ner_tokenizer(splitting, truncation=True, is_split_into_words=True, max_length=512)
    print('Tokenized inputs : {}'.format(ner_tokenized_inputs)) if ner_recognizer_debug_mode else None

    word_ids = ner_tokenized_inputs.word_ids()
    print(word_ids) if ner_recognizer_debug_mode else None

    prediction, label, _ = ner_trainer.predict([ner_tokenized_inputs])
    print('Prediction : {}'.format(prediction)) if ner_recognizer_debug_mode else None
    print('------------------------') if ner_recognizer_debug_mode else None

    prediction = np.argmax(prediction, axis=2)
    print('Prediction from argmax: {}'.format(prediction)) if ner_recognizer_debug_mode else None
    ner_prediction_long = [label_list[p] for p in prediction[0]]
    print('NER Predition long : {}'.format(ner_prediction_long)) if ner_recognizer_debug_mode else None

    ner_prediction = []  # the entire list with ner tags
    ner_indexes = []

    for y in range(0, len(splitting)):
        try:
            ner_prediction.append(ner_prediction_long[word_ids.index(y)])
            if ner_prediction_long[word_ids.index(y)] != 'O' and not is_number(splitting[y]):
                ner_indexes.append(y)
        except:
            continue

    print('NER of sequence : {}'.format(ner_prediction)) if ner_recognizer_debug_mode else None
    print('NER ner_indexes : {}'.format(ner_indexes)) if ner_recognizer_debug_mode else None
    return ner_indexes




#Data Collator Class
class DataCollatorForWholeWordMaskCustomized(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>"""
    
    global_mlm : bool = True
    global_mlm_prob : float=0.0
    ne_mlm_prob : float = 0.0
    verb_mlm_prob : float = 0.0
    noun_mlm_prob : float = 0.0

    #initilizer
    def __init__(self, tokenizer, global_mlm, global_mlm_prob, ne_mlm_prob, verb_mlm_prob, noun_mlm_prob, ner_tokenizer, ner_trainer, ne_pos_dict):
        super().__init__(tokenizer)
        self.global_mlm = global_mlm
        self.global_mlm_prob = global_mlm_prob
        self.ne_mlm_prob = ne_mlm_prob
        self.verb_mlm_prob = verb_mlm_prob
        self.noun_mlm_prob = noun_mlm_prob
        self.ner_tokenizer = ner_tokenizer
        self.ner_trainer = ner_trainer
        self.ne_pos_dict = ne_pos_dict


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        #checks instance of examples is a dicionary
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else: #list input
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # [AF-19/8] Code modification to be compatible with xlm-roberta-tokenizer
        cand_indexes = []
        sentence_words=[]
        for i, token in enumerate(input_tokens):
            if token == "<s>" or token == "</s>":
                continue

            if token.startswith("▁"):
                cand_indexes.append([i])
                sentence_words.append(token[1:])
            else:
                cand_indexes[-1].append(i)
                sentence_words[-1] = sentence_words[-1] + token

        sentence = ' '.join(sentence_words)
        print('[_whole_word_mask] DebugCheckpoint#0 sentence : {}'.format(sentence)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#1 input_tokens : {}'.format(input_tokens)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#2 cand_indexes : {}'.format(cand_indexes)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#2 sentence_words : {}'.format(sentence_words)) if whole_word_masking_debug_mode else None


        #word token check sent for NER model and continual pre-tranining
        assert len(sentence_words) ==len(cand_indexes)

        ne_token_indexes = []
        ne_cand_indexes = []

        if ne_mlm_prob > 0.0:
            # ner tokens and indexes
            # if directly getting ner_indexes from model
            # ner_token_indexes =  ne_recognizer(sentence_words, self.ner_tokenizer, self.ner_trainer)

            # from dict
            try:
                ne_token_indexes = list(ne_pos_dict[sentence]["ner_indexes"])
            except:
                # print('[Re-con. Sent] {}'.format(sentence))
                # print('[sentence_words][{}] {}'.format(len(sentence_words), sentence_words))
                ne_token_indexes = ne_recognizer(sentence_words, self.ner_tokenizer, self.ner_trainer)

            ne_cand_indexes = [cand_indexes[ner_index_list] for ner_index_list in ne_token_indexes]

        print('[_whole_word_mask] DebugCheckpoint#10 ner_token_indexes : {}\n'.format(ne_token_indexes)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#11 ne_cand_indexes : {}\n'.format(ne_cand_indexes)) if whole_word_masking_debug_mode else None

        #verb tokens and indexes
        verb_token_indexes = []
        verb_cand_indexes = []

        #noun tokens and indexes
        noun_token_idexes = []
        noun_cand_indexes = []

        try:
            if verb_mlm_prob >0:
                verb_token_indexes = ne_pos_dict[sentence]["verb_indexes"]
                verb_cand_indexes = [cand_indexes[index_list] for index_list in verb_token_indexes]

            if noun_mlm_prob > 0:
                noun_token_idexes = ne_pos_dict[sentence]["noun_indexes"]
                noun_cand_indexes = [cand_indexes[index_list] for index_list in noun_token_idexes]

        except:
            pass
            #print('[_whole_word_mask] DebugCheckpoint#11 EXCEPT for verb_indexes AND noun_indexes: {}\n'.format(sentence))


        print('[_whole_word_mask] DebugCheckpoint#11 verb_token_indexes : {}\n'.format(verb_token_indexes)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#11 verb_cand_indexes : {}\n'.format(verb_cand_indexes)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#11 noun_token_idexes : {}\n'.format(noun_token_idexes)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#11 noun_cand_indexes : {}\n'.format(noun_cand_indexes)) if whole_word_masking_debug_mode else None

        random.shuffle(cand_indexes)
        print('[_whole_word_mask] DebugCheckpoint#3 shuffled_cand_indexes : {}'.format(cand_indexes)) if whole_word_masking_debug_mode else None
        #total identified tokens
        total_ne_tokens = sum([len(item) for item in ne_cand_indexes])
        total_verb_tokens = sum([len(item) for item in verb_cand_indexes])
        total_noun_tokens = sum([len(item) for item in noun_cand_indexes])



        #calculate prediction tokens
        ne_num_to_predict = min(max_predictions, max(1, int(round(total_ne_tokens * self.ne_mlm_prob)))) if ne_mlm_prob >0 and len(ne_token_indexes) > 0 else 0
        verb_num_to_predict = min(max_predictions, max(1, int(round(total_verb_tokens * self.verb_mlm_prob))))  if verb_mlm_prob > 0 and len(verb_token_indexes) > 0 else 0
        noun_num_to_predict = min(max_predictions, max(1, int(round(total_noun_tokens * self.noun_mlm_prob))))  if noun_mlm_prob > 0 and len(noun_token_idexes) > 0 else 0
        global_num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.global_mlm_prob)))) if global_mlm_prob > 0 and  global_mlm else ne_num_to_predict + verb_num_to_predict + noun_num_to_predict

        # if global_num_to_predict < ne_num_to_predict + verb_num_to_predict + noun_num_to_predict :
        #     global_num_to_predict = ne_num_to_predict + verb_num_to_predict + noun_num_to_predict
        # else:
        #     random_tokens_to_predit = global_num_to_predict - (ne_num_to_predict + verb_num_to_predict + noun_num_to_predict)


        print('[_whole_word_mask] DebugCheckpoint#13 ne_num_to_predict = min(max_predictions, max(1, int(round({} * {})))) : {}'.format(total_ne_tokens, self.ne_mlm_prob, ne_num_to_predict)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#14 verb_num_to_predict = min(max_predictions, max(1, int(round({} * {})))) : {}'.format(total_verb_tokens, self.verb_mlm_prob, verb_num_to_predict)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#15 noun_num_to_predict = min(max_predictions, max(1, int(round({}*{})))) : {}'.format(total_noun_tokens, self.noun_mlm_prob, noun_num_to_predict)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#16 global_num_to_predict = min({}, int(round({}*{}))): {}'.format(max_predictions, len(input_tokens), self.global_mlm_prob, global_num_to_predict)) if whole_word_masking_debug_mode else None
        #print('[_whole_word_mask] DebugCheckpoint#16 random_num_to_predict : {}'.format(random_num_to_predict)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#17 global_num_to_predict : {}'.format(global_num_to_predict)) if whole_word_masking_debug_mode else None


        masked_lms = []
        covered_indexes = set() #add the indexes to masked in ths covered_indexes set.

        count_ne=0
        count_verb=0
        count_noun=0
        count_random=0

        if global_mlm:
            print("------------------------------------------------------------------------------------------------------------------------------")  if whole_word_masking_debug_mode else None
            for item_list_index in cand_indexes:

                # if already the number to predict has been identified for masking
                if len(masked_lms) >= global_num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.

                #enable for span masking
                # if len(masked_lms) + len(item_list_index) > global_num_to_predict:
                #     continue

                is_any_index_covered = False
                if item_list_index in ne_cand_indexes and count_ne < ne_num_to_predict:
                    count_ne+=1
                    covered_indexes, masked_lms = add_subword_from_item_list_indexes_to_masked_lms(item_list_index, covered_indexes, masked_lms)
                elif item_list_index in verb_cand_indexes and count_verb < verb_num_to_predict:
                    count_verb+=1
                    covered_indexes, masked_lms = add_subword_from_item_list_indexes_to_masked_lms(item_list_index, covered_indexes, masked_lms)
                elif item_list_index in noun_cand_indexes and count_noun < noun_num_to_predict:
                    count_noun+=1
                    covered_indexes, masked_lms = add_subword_from_item_list_indexes_to_masked_lms(item_list_index, covered_indexes, masked_lms)

                print('[{}] count_ne:{} count_verb:{} count_noun:{} count_random:{} masked_lms:{}'.format(item_list_index, count_ne, count_verb, count_noun, count_random, masked_lms))  if whole_word_masking_debug_mode else None

        random_tokens_to_predit = global_num_to_predict - count_ne - count_verb - count_noun if global_mlm and (global_num_to_predict - count_ne - count_verb - count_noun) > 0 else 0


        if global_mlm:
            for item_list_index in cand_indexes:
                # if already the number to predict has been identified for masking
                if len(masked_lms) >= global_num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.

                #enable for span masking
                # if len(masked_lms) + len(item_list_index) > global_num_to_predict:
                #     continue

                if count_random < random_tokens_to_predit:
                    count_random += 1
                    covered_indexes, masked_lms = add_subword_from_item_list_indexes_to_masked_lms(item_list_index, covered_indexes, masked_lms)
                print(
                    '[{}] count_ne:{} count_verb:{} count_noun:{} count_random:{} masked_lms:{}'.format(item_list_index,
                                                                                                        count_ne,
                                                                                                        count_verb,
                                                                                                        count_noun,
                                                                                                        count_random,
                                                                                                        masked_lms)) if whole_word_masking_debug_mode else None

            print("--------------------------------------------------------------------------------------------------------------------------")  if whole_word_masking_debug_mode else None


        print('\n\n[_whole_word_mask] DebugCheckpoint#18 random_tokens_to_predit = {}-{}-{}-{} : {}'.format(global_num_to_predict,
                                                                                                                  count_ne,
                                                                                                                  count_verb,
                                                                                                                  count_noun,
                                                                                                                  random_tokens_to_predit)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#22 masked_lms : {}'.format(masked_lms)) if whole_word_masking_debug_mode else None
        print('[_whole_word_mask] DebugCheckpoint#23 covered_indexes : {}'.format(covered_indexes)) if whole_word_masking_debug_mode else None

        print('[_whole_word_mask] DebugCheckpoint#24 masked_lms, covered_indexes :\n{}\n{}\n'.format(masked_lms,
                                                                                                     covered_indexes)) if whole_word_masking_debug_mode else None

        # if len(covered_indexes) != len(masked_lms):
        #     raise ValueError("Length of covered_indexes is not equal to length of masked_lms.") if whole_word_masking_debug_mode else None
        #generate mask_labels vector containing the ids for masking
        mask_labels = [1 if i in masked_lms else 0 for i in range(len(input_tokens))]

        print('[_whole_word_mask] DebugCheckpoint#24 output list of mask_labels :\n{}\n'.format(mask_labels)) if whole_word_masking_debug_mode else None
        print('---------------------------------End of [_whole_word_mask]------------------------\n\n') if whole_word_masking_debug_mode else None
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        print('[torch_mask_tokens] DebugCheckpoint# indices_replaced : {}'.format(indices_replaced)) if torch_mask_tokens_debug_mode else None
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        print('[torch_mask_tokens] DebugCheckpoint# inputs[indices_replaced] : {}'.format(inputs[indices_replaced])) if torch_mask_tokens_debug_mode else None

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # #verification purpose.
        if print_final_masking_inputs:
            print('[torch_mask_tokens] DebugCheckpoint#inputs :')
            for row in inputs:
                print('[{}] {}'.format(len(row.tolist()), row.tolist()))

            print('\n[torch_mask_tokens] DebugCheckpoint#input_tokens :')
            for row in inputs:
                print('[{}] {}'.format(len(row.tolist()), [tokenizer._convert_id_to_token(id) for id in row.tolist()]))


            print('\n[torch_mask_tokens] DebugCheckpoint#labels :')
            for row in labels:
                print('[{}] {}'.format(len(row.tolist()), row.tolist()))

            print('\n[torch_mask_tokens] DebugCheckpoint#label_tokens :')
            for row in labels:
                print('[{}] {}'.format(len(row.tolist()), [tokenizer._convert_id_to_token(id) for id in row.tolist() if id>0]))

        print('[torch_mask_tokens] DebugCheckpoint#OUTPUT inputs, labels : \n{}\n{}'.format(inputs, labels)) if torch_mask_tokens_debug_mode else None
        print('---------------------------------End of [torch_mask_tokens]------------------------\n\n') if torch_mask_tokens_debug_mode else None
        return inputs, labels

#torch collate batch
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

#A dataset wrapper, that tokenizes the data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

#tokenization and datacollator
train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(valid_sentences, tokenizer, max_length, cache_tokenization=True) if len(valid_sentences) > 0 else None
wwm_custom_data_collator= DataCollatorForWholeWordMaskCustomized(tokenizer=tokenizer,
                                                             global_mlm=global_mlm,
                                                             global_mlm_prob=global_mlm_prob,
                                                             ne_mlm_prob=ne_mlm_prob,
                                                             verb_mlm_prob=verb_mlm_prob,
                                                             noun_mlm_prob=noun_mlm_prob,
                                                             ner_tokenizer = ner_tokenizer,
                                                             ner_trainer = ner_trainer,
                                                             ne_pos_dict=ne_pos_dict)

#TrainingArguments - https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#trainingarguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=5000,
    save_steps=save_steps,
    logging_steps=5000,
    save_total_limit=30,
    prediction_loss_only=True,
    fp16=use_fp16,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=wwm_custom_data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()
print(trainer.state.log_history)


print("Saved models to:", output_dir)

elapsedTime=time.time()-startTime

hours=int(elapsedTime//3600)
mints=int((elapsedTime%3600)//60)
secs=int((elapsedTime%3600)%60)
print('Elapsed time : {}hrs {}min {}sec'.format(hours, mints, secs))

print('Script completed.......')
print("Continual Pre-training completed")


