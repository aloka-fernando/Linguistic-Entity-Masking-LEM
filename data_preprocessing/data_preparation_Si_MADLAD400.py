#on GPU215
import string
import re
import fasttext
from  datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
madlad_sin = load_dataset("allenai/madlad-400", "si", split="clean")
 
sample_size =1000000
file_valid = open('valid-madlad400.dd.si.txt', 'r', encoding='utf8')
file_out= open('train{}K-madlad400.si.txt'.format(int(sample_size/1000)), 'w', encoding='utf8')
#file_out= open('train{}-madlad400.si.txt'.format("all"), 'w', encoding='utf8')

valid_set_sentences=set([line.strip() for line in file_valid])
sent_ending_suffixes=["...", "///", "???", "..", "//"]
bible_books={"උත්පත්ති","නික්මයාම","ලෙවී කථාව","ගණන් කථාව","ද්වීතීය කථාව","යෝෂුවා","විනිශ්චයකාරයන්ගේ පොත","රූත්ගේ කථාව","1 සාමුවෙල්","2 සාමුවෙල්","1 රාජාවලිය","2 රාජාවලිය","1 ලේකම්","2 ලේකම්","එස්රා","නෙහෙමියා","එස්තර්","යෝබ්","ගීතාවලිය","හිතෝපදේශ","දේශනාකාරයා","සාලමොන්ගේ ගීතිකා","යෙසායා","යෙරෙමියා","විලාප ගී","එසකියෙල්","දානියෙල්","හොෂෙයා","යෝවෙල්","ආමොස්","ඔබදියා","යෝනා","මීකා","නාහුම්","හබක්කුක්","ශෙපනියා","හග්ගයි","සෙකරියා","මලාකි","ශු. මතෙව්","ශු. මාර්ක්","ශු. ලූක්","ශු. යොහන්","ක්රිරයා","රෝම","1 කොරින්ති","2 කොරින්ති","ගලාති","එපීස","පිලිප්පි","කොලොස්සි","1 තෙසලෝනික","2 තෙසලෝනික","1 තිමෝති","2 තිමෝති","තීතස්","පිලෙමොන්","හෙබ්රෙනව්","යාකොබ්","1 පේතෘස්","2 පේතෘස්","1 යොහන්","2 යොහන්","3 යොහන්","යූදස්","එළිදරව්"}

#LID model
#sin_Sinh, tam_Taml, eng_Latn
LID='sin_Sinh'
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
#print(madlad_sin.info)
# Found cached dataset madlad-400 (/userdirs/aloka/.cache/huggingface/datasets/allenai___madlad-400/si/0.0.0/f1efa01435272063f0b25269ddfbd8a3f49279a030b155784129404874b3db34)
# DatasetInfo(description='MADLAD-400 (Multilingual Audited Dataset: Low-resource And Document-level) is a document-level multilingual dataset based on Common Crawl, covering 419 languages in total. We use all snapshots of CommonCrawl available as of August 1, 2022. The primary advantage of this dataset over similar datasets is that it is more multilingual (419 languages), it is audited and more highly filtered, and it is document-level. The main disadvantage is also its strength -- being more filtered, it may lack the recall needed for some applications.\n', citation='\n@misc{kudugunta2023madlad400,\n      title={MADLAD-400: A Multilingual And Document-Level Large Audited Dataset}, \n      author={Sneha Kudugunta and Isaac Caswell and Biao Zhang and Xavier Garcia and Christopher A. Choquette-Choo and Katherine Lee and Derrick Xin and Aditya Kusupati and Romi Stella and Ankur Bapna and Orhan Firat},\n      year={2023},\n      eprint={2309.04662},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}\n', homepage='', license='', features={'text': Value(dtype='string', id=None)}, post_processed=None, supervised_keys=None, task_templates=None, builder_name='madlad-400', config_name='si', version=0.0.0, splits={'clean': SplitInfo(name='clean', num_bytes=4868972843, num_examples=349220, shard_lengths=[36000, 37000, 36000, 36000, 37000, 36000, 36000, 36000, 36000, 23220], dataset_name='madlad-400'), 'noisy': SplitInfo(name='noisy', num_bytes=8380612236, num_examples=788048, shard_lengths=[47000, 47000, 48000, 48000, 48000, 47000, 47000, 48000, 48000, 48000, 47000, 48000, 47000, 48000, 47000, 47000, 28048], dataset_name='madlad-400')}, download_checksums={'https://huggingface.co/datasets/allenai/MADLAD-400/resolve/ecd71297d60c1eb996cd3d7c44c60ad5b55adfc6/data/si/si_clean_0000.jsonl.gz': {'num_bytes': 1073299556, 'checksum': None}, 'https://huggingface.co/datasets/allenai/MADLAD-400/resolve/ecd71297d60c1eb996cd3d7c44c60ad5b55adfc6/data/si/si_noisy_0000.jsonl.gz': {'num_bytes': 1883326811, 'checksum': None}}, download_size=2956626367, post_processing_size=None, dataset_size=13249585079, size_in_bytes=16206211446)


#print first example
print(madlad_sin[0]["text"])
#{'text': 'දැන් අපේ වගකීම මුළු මනුෂ්\u200dය සංහතිය ම වෙනුවෙන් | Sinhala story Blog\\nදැන් අපේ වගකීම මුළු මනුෂ්\u200dය සංහතිය ම වෙනුවෙන්\\nග්ලෝරියානාට ඒ කතාව හරි හැටියකට වැටහුනේ නැත.\\nඇයගේ හැදියාව ප්\u200dරායෝගික වූවකි. සිද්ධාන්තයක් ගැන නොවූවකි. ඒ නිසා ටලී කියූ ජයග්\u200dරහණයකින් ඇති වන්නා වූ වගකීම් ගැන අදහස් ඇයගේ සිත නොසන්සුන් කරන ලදි.\\n“ඔයා මේ හදන්නෙ අපි ඇමෙරිකාව ප්\u200dරතිසංස්කරණය කරන්න ඕනා කියලා කියන්න නෙමෙයි කියලා මම බලාපොරොත්තු වෙනවා,” ඇය කීවාය, “මොකද එහෙම නම්, අපිට එතකොට එයාලගෙන් සල්ලි ඉල්ලා ගන්න වෙනවා ඒක කරන්නට.”\\n“නැහැ,” ඔහු පිළිතුරු දුන්නේ ය. “ග්\u200dරෑන්ඩ් ෆෙන්වික් දනව්වේ අපි විසිවැනි සියවසේ හමුදා හාස්කම සාක්ෂාත් කරලා -ජයගත්තාට පැරදුන අයට සතයක් වත් දිය යුතු නැති යුද්ධයක් ජයගත්තා. ඒත් අපි ලේසියෙන් නිදහස් වෙන්නෙ නැහැ. අපි යුද්ධයට යන්නට පෙර අපේ වගකීම වූයේ අපි පමණයි. දැන් අපේ වගකීම මුළු මනුෂ්\u200dය සංහතිය ම වෙනුවෙන්. මේ බෝම්බය පාවිච්චි කරන්නට හදන අයගේ දෑත් වලින් අපි ඒක අයින් කරලා තියාගන්න ඕන. මොකද මේ වගේ බෝම්බයක් පාවිච්චි කරන්න සිද්ධ වෙන යුද්ධයක දී ඒකීය වූ ජාතීන් නෙමෙයි වඳ වෙලා යන්නෙ සියළු මනුෂ්\u200dය සංහතියයි.\\n“ඒක ලෙහෙසි නැහැ, මම දන්නවා, අපි වගේ පුංචි රටක් වෙලා ඉඳලා, මේ වගේ ටික කාලයක දී ලෝකයේ බාරකරුවා වෙන්නට සිද්ධ වීම. මීට වඩා ලොකු රටවල් වලට මේ බාරදූර කාර්යය දරා ගන්න අපහසු වෙලා තියෙනවා. දෙවැනි ලෝක යුද්ධයෙන් පස්සෙ ඇමෙරිකාව හදිසියේ ම ලෝකයේ පළමුවැනි රට බවට පත්වූවා. ඊට පස්සෙ ඒ ගැන මොකද කරන්නෙ කියලා කරකියා ගන්න බැරිව හිටියා. වැඩි කොටසක් කැමති වුනේ නැහැ ලෝකයේ ප්\u200dරමුඛයෝ වෙන්නට. ආපහු හිටපු විදියට යන්න ඕනෑ යැයි කියා සිටියා. ඒත් ඒක එදා ඔවුන්ට කළ නොහැකි වූවක්. අද අපිට කරත නොහැකි වූවක් වගේමයි.”\\n“ඇමෙරිකානුවන් අපිට මිලියන මිලියන ගණන් වලින් ඩොලර් දෙනවා නම් එතකොට ඔබ හිතන්නෙ නැද්ද අපි බෝම්බය ආපහු එයාලට ම බාර දිය යුතු යැයි කියා?”\\nටලී හිටගත්තේ ය. දෙවුර කෙළින් කර ගත්තේ ය. තම අත පළල් සියපතේ හිසෙහි රැඳවූයේය. ඔහුගේ ඉරියව්ව ග්ලෝරියානාට නැවතත් සර් රොජර් ෆෙන්වික් සිහිගන්වන්නට සමත් විය.\\n“අපිට පැවරිලා තියෙනවා,” ටලී ගම්භීරව කියා සිටියේය, “වගකීමක්, මේ බෝම්බය අපට ලැබීම පමණකින් ලැබුණා වූවක්. ඒ වගකීම තමයි මුළු ලෝකය ම නැවත පියවි තත්වයට ගෙනෙන්නට. ඔබ, ආදිපාදවරියෙනි, දැන් මේ කුඩා ග්\u200dරෑන්ඩ් ෆෙන්වික් දනව්වේ නායකයා පමණක් නෙමෙයි. ඔබ මුළු ලෝකයේ ම බලවත් කාන්තාව. මිලියන ගණනකගේ ජීවිත ඔබේ වදන් වලට යටත්. ඔබ මට අණ දුන් පමණින් මම මේ බෝම්බය මගේ මුගුරෙන් ගහන එක පහරින් පුපුරුවා හැරලා මුළු යුරෝපය ම විනාශ කරලා දාන්න පුළුවන්. ඒ වගේ සුවිසල් බලයක් සමඟ හැම ජාතිකයකට ම ඔබ හා ගිවිසුම් ගහන්නට සිද්ධ වෙනවා. ලෝක සාමය ගැන වූ ගිවිසුම්. ඒවා ප්\u200dරබල ගිවිසුම්. මොකද මේ බෝම්බය පුපුරුවා හරින තර්ජනයෙන් ඒවා සාක්ෂාත් කරගන්න පුළුවන්. ඔබේ පරම්පරාවේ මුතුන් මිත්තන් කවදා වත් තමන්ගේ ජාතිය වෙනුවෙන් වෙහෙසීම මඟ හැරියේ නැහැ. ඒත් අද ඔබව කැඳවලා තියෙන්නෙ මුළු ලෝකය ම වෙනුවෙන් වෙහෙසීම මඟ නොහරින්නට.”\\n“ඒත්, ඉතින් අනිත් ජාතීන් ද කාලයාගේ ඇවෑමෙන් මේ බෝම්බය ම හදාවි නේද?” ග්ලෝරියානා ඇසුවා ය.\\n“ඇත්ත,” ටලී උත්තර දුනි, “ඒ අයට බෝම්බ හදන්නට නොහැකියාව ඇති කරන්නේ කෙසේ ද යන ගැටළුවට තමයි අපි උත්සාහ ගත යුත්තේ. මේ කාරණයේ හැම පැත්ත ම සලකා බලන්නට පරෙවි මඩුල්ල කැඳවිය යුතු යැයි මම යෝජනා කරනවා.”\\nටලී හා වූ පුද්ගලික හමුවකින් පසුව මවුන්ට්ජෝයි සිටුවරයා බෝම්බය ග්\u200dරෑන්ඩ් ෆෙන්වික් වෙත ගෙන ඒම ගැන ටලීව ද්යෝෂාභියෝගයට ලක් කර දනව්වෙන් නෙරපා හරින්නට නැවතත් දැඩි සේ කියා සිටියේය. ඔහුගේ යෝජනාව පැත්තකට දැමූ ග්ලෝරියානා ආදිපාදවරිය ඊ ළඟ දිනයට පරෙවි මඩුල්ල කැඳ වූවාය.'}

#num_examples=349220,
print('Dataset size : {}'.format(len(madlad_sin)))

#flag
hasReachedLimit=False

count=0
used_sentences=set()
#print Si Text
for madlad_sin_example in madlad_sin:
    lines = madlad_sin_example["text"].split('\\n')[2:] #Remove repetition words දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන් | Sinhala story Blog (2)දැන් අපේ වගකීම මුළු මනුෂ්‍ය සංහතිය ම වෙනුවෙන්

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
            #sentence = sentence.replace("\u0dca\u0dbb", "\u0DCA\u200D\u0dbb") #දුම්රිය> දුම්‍රිය
            # line=line.replace("\u0dca\u0020\u0dba", "\u0DCA\u200D\u0dba") #මෙන්ය > මෙන්ya
            file_out.write('{}\n'.format(sentence))
            count+=1

        if count >= sample_size:
            hasReachedLimit=True
            break

    if hasReachedLimit:
        break

print('Sucessfully created')
file_out.close()


