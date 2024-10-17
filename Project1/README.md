## Web Searching & Mining Project 1
##### Author : Dept. of Mathematical Science, Yi-Shan Chu 應數三 朱翊杉 ID:111701037
---

### Environment Setup

The original platform is Ubuntu 24.04 LTS
 CPU: i5-9400F
 RAM: 64GB DDR4
 GPU: RTX 4080 + RTX 2060Super
Python >= 3.11.10

If you have the problems with nltk, remove the `#` in Line 22
```
#nltk.download('all')
```
If you are using conda, create a new env
```conda create -n <name> python=3.11```

``` pip install -r requirement.txt ```


### Command
```
python main.py --Eng_query <your english query here> --Chi_query <your chinese query here>
```
For example with query "Typhoon Taiwan war" and "資安 遊戲"
```
python main.py --Eng_query "Typhoon Taiwan war" --Chi_query "資安 遊戲"
```

### File Structure
./ChineseNews
    Newsxxxx.txt
./EnglishNews
    Newsxxxx.txt
./smaller_dataset
    ./collections
    ./queries
    rel.tsv
EnglishStopwords.txt -- 英文停用詞
ChineseStopwords.txt -- 中文停用詞
main.py -- 主要執行檔案
metrics.py -- 計算MRR、MAP、Recall at 10.
tools.py -- Some useful functions.
tfidfcalculation.py -- calculate the TF vector, IDF vector, TF-IDF vector. In TF weighting calculation, Maximum frequency normalization is employed refer to the course slide.
ranks.py -- Functions to output the results.

---
If you got any issues when using, please don't hesitate to contact ```111701037@g.nccu.edu.tw```