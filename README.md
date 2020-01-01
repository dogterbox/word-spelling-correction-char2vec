# Words Spelling Correction using Char2Vec
**Medium blog: https://medium.com/@dogterbox/ลองทำ-word-spelling-correction-ด้วย-char2vec-6167430492bc**

## Data Preparing

Load corpus: **PyThaiNLP** thai word corpus


```python
from pythainlp.corpus import thai_words
import numpy as np

words = thai_words()
words = np.array(list(words))  # to array
```


```python
print('Number of vocabs: ', len(words))
```

    Number of vocabs:  62847



```python
print(words[:10])  # sample
```

    ['กระจาว' 'ลูกเมียหลวง' 'ปุลินท์' 'ล่วงเวลา' 'คุยหรหัสย์' 'คู่สามีภรรยา'
     'มรสุม' 'ดีนาคราช' 'จานเจือ' 'ข้อบังคับของบริษัท']


Convert array to string separate by '\n'


```python
words_str = '\n'.join(words)
```


```python
print(words_str[:100])  # sample
```

    กระจาว
    ลูกเมียหลวง
    ปุลินท์
    ล่วงเวลา
    คุยหรหัสย์
    คู่สามีภรรยา
    มรสุม
    ดีนาคราช
    จานเจือ
    ข้อบังคับของบริษั


Convert string to char


```python
words_char = list(words_str)
```


```python
print(words_char[:100])  # sample
```

    ['ก', 'ร', 'ะ', 'จ', 'า', 'ว', '\n', 'ล', 'ู', 'ก', 'เ', 'ม', 'ี', 'ย', 'ห', 'ล', 'ว', 'ง', '\n', 'ป', 'ุ', 'ล', 'ิ', 'น', 'ท', '์', '\n', 'ล', '่', 'ว', 'ง', 'เ', 'ว', 'ล', 'า', '\n', 'ค', 'ุ', 'ย', 'ห', 'ร', 'ห', 'ั', 'ส', 'ย', '์', '\n', 'ค', 'ู', '่', 'ส', 'า', 'ม', 'ี', 'ภ', 'ร', 'ร', 'ย', 'า', '\n', 'ม', 'ร', 'ส', 'ุ', 'ม', '\n', 'ด', 'ี', 'น', 'า', 'ค', 'ร', 'า', 'ช', '\n', 'จ', 'า', 'น', 'เ', 'จ', 'ื', 'อ', '\n', 'ข', '้', 'อ', 'บ', 'ั', 'ง', 'ค', 'ั', 'บ', 'ข', 'อ', 'ง', 'บ', 'ร', 'ิ', 'ษ', 'ั']


Save data to **words-char.txt** file


```python
with open('words-char.txt', mode='w', encoding='utf-8') as file:
    file.write(' '.join(words_char))
```

## Char2Vec Model
Using FastText traing Word2Vec model on **character level**


```python
import fasttext
```

Load file and model training


```python
model = fasttext.train_unsupervised('words-char.txt', epoch=200, ws=3)
```


```python
print('Number of char: ', len(model.words))
```

    Number of char:  73


Create word vector on character level model


```python
words_vec = [model.get_sentence_vector(' '.join(list(word))) for word in words]
words_vec = np.array(words_vec)
```

## Word Spelling Correction
Using **nearest neighbors model** to find **word similarity** for get word suggestion

Model training
- X: word vector
- y: word/vocab


```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

X, y = words_vec, words
nbrs = NearestNeighbors().fit(X, y)
```

### Save All Model


```python
import joblib

model.save_model('char2vec.bin')  # fasttext model
joblib.dump(words, 'words.joblib')
joblib.dump(nbrs, 'nbrs.joblib');
```

# Usage


```python
import fasttext
import joblib

model = fasttext.load_model('char2vec.bin')
words = joblib.load('words.joblib')
nbrs = joblib.load('nbrs.joblib')
```

    



```python
words_input = ['การบด้าน', 'สวัดี', 'vออกเลอร์', 'ปละเทศไทยย', 'อรอย']
```


```python
word_input_vec = [model.get_sentence_vector(' '.join(list(word))) for word in words_input]
indices = nbrs.kneighbors(word_input_vec, 5, False)  # n_neighbors is 5
suggestion = words[indices]

for w, s in zip(words_input, suggestion):
    print(f'{w} \n---> {s}')
```

    การบด้าน 
    ---> ['การบ้าน' 'หยาบกร้าน' 'หน้ากระดาน' 'การอาบน้ำ' 'กลับด้าน']
    สวัดี 
    ---> ['สวัสดี' 'วสวัดดี' 'วิดัสดี' 'ดบัสวี' 'วนัสบดี']
    vออกเลอร์ 
    ---> ['ล็อกเกอร์' 'เดอะรอยัลกอล์ฟ' 'ล็อคเกอร์' 'ออยเลอร์' 'อาร์เซนอล']
    ปละเทศไทยย 
    ---> ['ประเทศไทย' 'การปิโตรเลียมแห่งประเทศไทย' 'ไปรษณีย์ลงทะเบียน' 'ลายเทศ'
     'ประเทศอินเดีย']
    อรอย 
    ---> ['รอย' 'อร่อย' 'รอคอย' 'รอยต่อ' 'ร่องรอย']



```python
!jupyter nbconvert --to markdown char2vec_spelling_correction.ipynb
```

    [NbConvertApp] Converting notebook char2vec_spelling_correction.ipynb to markdown
    [NbConvertApp] Writing 3587 bytes to char2vec_spelling_correction.md



```python

```
