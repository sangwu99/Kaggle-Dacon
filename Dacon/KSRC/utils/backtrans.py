from pororo import pororo 
from glob import glob

nmt = Pororo(task="translation", lang="multi")

def make_aug_dataset(data, src_lang, tgt_lang, types):
    kor_train_list = []
    difflang_train_list = []

    for i, text in enumerate(tqdm(data[types])):
        
        difflang = nmt(src=src_lang, text=text, tgt=tgt_lang)
        ko = nmt(src=tgt_lang, text=difflang, tgt=src_lang)

        difflang_train_list.append(difflang)
        kor_train_list.append(ko)
#         print(f"[{i}, \"{text}\" \n, \"{difflang}\", \n \"{ko}\"],")
    
    return kor_train_list, difflang_train_list

