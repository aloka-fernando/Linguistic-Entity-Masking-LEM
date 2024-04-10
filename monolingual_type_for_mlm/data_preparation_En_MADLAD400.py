#on GPU215
import string
import re
import fasttext
from  datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk

#tokenizer and dataset
nltk.download('punkt')
madlad_en = load_dataset("allenai/madlad-400", "en", split="clean", streaming=True)
#downloaded to madlad-400/en to /userdirs/aloka/.cache/huggingface/datasets/allenai

#parameters
sample_size =1000000
file_valid = open('valid-madlad400.dd.en.txt', 'r', encoding='utf8')
file_out = open('train{}K-madlad400.en.txt'.format(int(sample_size/1000)), 'w', encoding='utf8')
#file_out= open('train{}K-madlad400.en.txt'.format("-all"), 'w', encoding='utf8')

valid_set_sentences=set([line.strip() for line in file_valid])
sent_ending_suffixes=["...", "///", "???", "..", "//"]
bible_books={"Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges","Ruth","1 Samuel","2 Samuel","1 Kings","2 Kings","1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job","Psalms","Proverbs","Ecclesiastes","Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel","Hosea","Joel","Amos","Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai","Zechariah","Malachi","Matthew","Mark","Luke","John","Acts","Romans","1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians","1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon","Hebrews","James","1 Peter","2 Peter","1 John","2 John","3 John","Jude","Revelation"}
words_to_exclude ={"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","bedroom", "feet", "[…]","<BR>-", "I","RTI","January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December","comments", "AM", "PM","more","facilities","room","Posted","ahhhh",":D"}

#LID model
#sin_Sinh, tam_Taml, eng_Latn
LID='eng_Latn'
pretrained_lang_model = "nllblid218e"
model = fasttext.load_model('/userdirs/aloka/pre-trained-models/{}'.format(pretrained_lang_model))

#regexp pattern
url_pattern = r'https?://\S+\.html'
num_prefix_pattern=r'^\(\d+\)\s|^\d+\.|^\d+\:'
date_pattern = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|(?:\d{2}/){2}\d{4})\b")

def get_lid(text):
    predictions = model.predict(text, k=1)
    lang_code = predictions[0][0].strip().split('__')[-1]
    #prob = predictions[1][0]
    return lang_code

#print - dataset summary
#print(madlad_en.info)
# Found cached dataset madlad-400 (/userdirs/aloka/.cache/huggingface/datasets/allenai___madlad-400/si/0.0.0/f1efa01435272063f0b25269ddfbd8a3f49279a030b155784129404874b3db34)
# DatasetInfo(description='MADLAD-400 (Multilingual Audited Dataset: Low-resource And Document-level) is a document-level multilingual dataset based on Common Crawl, covering 419 languages in total. We use all snapshots of CommonCrawl available as of August 1, 2022. The primary advantage of this dataset over similar datasets is that it is more multilingual (419 languages), it is audited and more highly filtered, and it is document-level. The main disadvantage is also its strength -- being more filtered, it may lack the recall needed for some applications.\n', citation='\n@misc{kudugunta2023madlad400,\n      title={MADLAD-400: A Multilingual And Document-Level Large Audited Dataset}, \n      author={Sneha Kudugunta and Isaac Caswell and Biao Zhang and Xavier Garcia and Christopher A. Choquette-Choo and Katherine Lee and Derrick Xin and Aditya Kusupati and Romi Stella and Ankur Bapna and Orhan Firat},\n      year={2023},\n      eprint={2309.04662},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}\n', homepage='', license='', features={'text': Value(dtype='string', id=None)}, post_processed=None, supervised_keys=None, task_templates=None, builder_name='madlad-400', config_name='si', version=0.0.0, splits={'clean': SplitInfo(name='clean', num_bytes=4868972843, num_examples=349220, shard_lengths=[36000, 37000, 36000, 36000, 37000, 36000, 36000, 36000, 36000, 23220], dataset_name='madlad-400'), 'noisy': SplitInfo(name='noisy', num_bytes=8380612236, num_examples=788048, shard_lengths=[47000, 47000, 48000, 48000, 48000, 47000, 47000, 48000, 48000, 48000, 47000, 48000, 47000, 48000, 47000, 47000, 28048], dataset_name='madlad-400')}, download_checksums={'https://huggingface.co/datasets/allenai/MADLAD-400/resolve/ecd71297d60c1eb996cd3d7c44c60ad5b55adfc6/data/si/si_clean_0000.jsonl.gz': {'num_bytes': 1073299556, 'checksum': None}, 'https://huggingface.co/datasets/allenai/MADLAD-400/resolve/ecd71297d60c1eb996cd3d7c44c60ad5b55adfc6/data/si/si_noisy_0000.jsonl.gz': {'num_bytes': 1883326811, 'checksum': None}}, download_size=2956626367, post_processing_size=None, dataset_size=13249585079, size_in_bytes=16206211446)


#flag
hasReachedLimit=False

count=0
used_sentences=set()
#print Si Text
for madlad_sin_example in madlad_en:
    lines = madlad_sin_example["text"].split('\\n')[2:] #Remove repetition words දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන් | Sinhala story Blog (2)දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන්

    print('Total sentences : {}'.format(count))
    for line in lines:
        line = re.sub(url_pattern, ' ', line)  # replace urls within text
        sentences=sent_tokenize(line)

        # sentence filtration
        sentences = [s for s in sentences if len(s.split()) > 6 and get_lid(s) == LID and s[-3:] not in sent_ending_suffixes and s[-2:] not in sent_ending_suffixes and len(bible_books.intersection(set(s.split())))==0 and len(words_to_exclude.intersection(set(s.split())))==0 and not (re.search("<p>", s)) and not re.search("<BR>", s) and not date_pattern.search(s)]

        for sentence in sentences:
            sentence = re.sub(num_prefix_pattern, '', sentence).strip('~').strip('"').strip()
            #sentence = sentence.replace("\u0dca\u0dbb", "\u0DCA\u200D\u0dbb") #දුම්රිය> දුම්‍රිය
            # line=line.replace("\u0dca\u0020\u0dba", "\u0DCA\u200D\u0dba") #මෙන්ය > මෙන්ya

            if len(used_sentences.intersection([sentence]))==0 and len(valid_set_sentences.intersection([sentence]))==0:
                used_sentences.add(sentence)
                file_out.write('{}\n'.format(sentence))
                count+=1

        if count >= sample_size:
            hasReachedLimit=True
            break

    if hasReachedLimit:
        break

print('Sucessfully created')
file_out.close()


