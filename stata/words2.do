cscript

version 16

python:
import nltk   
import requests
from bs4 import BeautifulSoup


url = "http://www.uone-tech.cn/news/stata16.html"    
html = requests.get(url) 
html.encoding='utf-8' 
text = BeautifulSoup(html.text).get_text() 
print(text)

import jieba       
words = jieba.lcut(text)        
counts ={}            
for word in words:
    word = word.strip()
    if len(word) <=1:            
        continue
    else:
        counts[word] = counts.get(word,0)+1          

items = list(counts.items())
items.sort(key = lambda x:x[1],reverse = True)

words = {}
for i in range(100):
    word,count = items[i]    
    words[word] = count

font = r'C:\Windows\Fonts\simsun.ttc'
wordcloud = WordCloud(max_font_size=75, font_path=font, max_words=100, background_color="white").generate_from_frequencies(words)

from sfi import Platform
import matplotlib
if Platform.isWindows():
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("words2.png")

end
