import string
import re
import fasttext
from  datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
madlad_ds = load_dataset("allenai/madlad-400", "ta", split="clean")
sample_size =150000
file_valid = open('valid-madlad400.dd.ta.txt', 'r', encoding='utf8')
file_out= open('train{}K-madlad400.ta.txt'.format(int(sample_size/1000)), 'w', encoding='utf8')
#file_out= open('train{}-madlad400.ta.txt'.format("-all"), 'w', encoding='utf8')

valid_set_sentences=set([line.strip() for line in file_valid])
sent_ending_suffixes=["...", "///", "???", "..", "//"]
bible_books={"ஆதியாகமம்","யாதிராகமம்","லேவியராகமம்","எண்ணாகமம்","உபாகமம்","யோசுவா","நியாயாதிபதிகள்","ரூத்","1 சாமுவேல்","2 சாமுவேல்","1 அராமுக்கு எழுந்த ராஜாவின் வரலாறு","2 அராமுக்கு எழுந்த ராஜாவின் வரலாறு","1 நாளன்","2 நாளன்","எஸ்றா","நெகேமியா","எஸ்தர்","யோபு","சங்கீதம்","நீதிமொழிகள்","உன்னத பரமபத உரை","உன்னத பரமபத பரிசுத்தம்","ஏசாயா","எரேமியா","புலமையின் வாழ்வின் சரித்திரம்","எசேக்கியேல்","தானியேல்","ஓசியா","யோவேல்","ஆமோஸ்","ஒபதியா","யோனா","மீகா","நாகூம்","ஆபகூக்","செப்பனியா","ஏக்காரியா","சகரியா","மலாக்கி","மத்தேயு","மார்க்","லூக்கா","யோவான்","அப்போஸ்தலருடைய நாயகம்","ரோமர்","1 கொரிந்தியர்","2 கொரிந்தியர்","கலாத்தியர்","எபேசியர்","பிலிப்பியர்","கொலோசெயர்","1 தெசலோனிக்கேயர்","2 தெசலோனிக்கேயர்","1 தீமோத்தேயு","2 தீமோத்தேயு","தீது","பிலேமோன்","எபிரேயர்","யாகோபு","1 பேதுரு","2 பேதுரு","1 யோவான்","2 யோவான்","3 யோவான்","யூதா","வேளாராயுதின் உரை"}

#LID model
#sin_Sinh, tam_Taml, eng_Latn
LID='tam_Taml'
pretrained_lang_model = "nllblid218e"
model = fasttext.load_model('/userdirs/aloka/pre-trained-models/{}'.format(pretrained_lang_model))

#regexp pattern
url_pattern = r'https?://\S+\.html'
num_prefix_pattern=r'^\(\d+\)\s|^\d+\.|^\d+\:'

def get_lid(text):
    predictions = model.predict(text, k=1)
    lang_code = predictions[0][0].strip().split('__')[-1]
    #prob = predictions[1][0]
    return lang_code

#print - dataset summary
#print(madlad_ds.info)

#print first example
print(madlad_ds[0]["text"])

#num_examples
print('Dataset size : {}'.format(len(madlad_ds)))

#flag
hasReachedLimit=False

count=0
used_sentences=set()
#print Ta Text
for madlad_example in madlad_ds:
    lines = madlad_example["text"].split('\\n')[2:] #Remove repetition words දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන් | Sinhala story Blog (2)දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන්

    print('Total sentences : {}'.format(count))
    for line in lines:
        line = re.sub(url_pattern, ' ', line)  # replace urls within text
        sentences=sent_tokenize(line)
        # sentence filtration
        sentences = [s for s in sentences if len(s.split()) > 6 and get_lid(s) == LID and s[-3:] not in sent_ending_suffixes and s[-2:] not in sent_ending_suffixes and len(bible_books.intersection(set(s.split())))==0]

        for sentence in sentences:
            sentence = re.sub(num_prefix_pattern, '', sentence).strip().strip('"')

            if len(used_sentences.intersection([sentence])) == 0 and len(valid_set_sentences.intersection([sentence])) == 0:
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


