# Stata Python 融合应用

##  [Hua Peng@StataCorp][hpeng]
### 2019 Stata 中国用户大会
### [https://huapeng01016.github.io/china2019/](https://huapeng01016.github.io/china2019/)

# Stata 16与Python的紧密结合

- 嵌入与运行Python程序
- 互动式执行Python
- 在do-file中定义与运行Python程序
- 在ado-file中定义与运行Python程序
- Python与Stata通过Stata Function Interface (sfi)互动

# 互动式执行Python

## **Hello World!**

~~~~
<<dd_do>>
python:
print('Hello World!')
end
<</dd_do>>
~~~~

## **for** 循环

Stata与其他Python环境一样，输入Python语句需要正确使用“缩进”。

~~~~
<<dd_do>>
python:
sum = 0
for i in range(7):
    sum = sum + i
print(sum)
end
<</dd_do>>
~~~~

## **sfi**

~~~~
<<dd_do>>
python:
from functools import reduce
from sfi import Data, Macro

stata: quietly sysuse auto, clear

sum = reduce((lambda x, y: x + y), Data.get(var='price'))

Macro.setLocal('sum', str(sum))
end
display "sum of var price is : `sum'"
<</dd_do>>
~~~~

## 更多**sfi**

~~~~
<<dd_do>>
python:
sum1 = reduce((lambda x, y: x + y), Data.get(var='rep78'))
sum1
sum2 = reduce((lambda x, y: x + y), Data.get(var='rep78', selectvar=-1))
sum2
end
<</dd_do>>
~~~~


# 使用Python模块

* Pandas
* Numpy
* BeautifulSoup, lxml
* Matplotlib
* Scikit-Learn, Tensorflow, Keras
* NLTK,jieba

# 三维曲面图

## 导入Python模块

~~~~
<<dd_do>>
python:
import numpy as np
from sfi import Platform

import matplotlib
if Platform.isWindows():
	matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sfi import Data
end
<</dd_do>>
~~~~

## 使用**sfi.Data**导入数据

~~~~
<<dd_do>>
use https://www.stata-press.com/data/r16/sandstone, clear
* Use sfi to get data from Stata
python:
D = np.array(Data.get("northing easting depth"))
end
<</dd_do>>
~~~~

## 使用三角网画图

~~~~
python:
ax = plt.axes(projection='3d')
ax.xaxis.xticks(np.arange(60000, 90001, step=10000))
ax.yaxis.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap='viridis', edgecolor='none')
plt.savefig("sandstone.png")
end
~~~~

<<dd_do:quietly>>
python:
ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2],
	cmap='viridis', edgecolor='none')
plt.savefig("sandstone.png")
end
<</dd_do>>

## 结果

![sandstone.png](./sandstone.png "sandstone.png")


## 改变颜色和视角

~~~~
python:
ax.plot_trisurf(D[:,0], D[:,1], D[:,2],
	cmap=plt.cm.Spectral, edgecolor='none')
ax.view_init(30, 60)
plt.savefig("sandstone1.png")
end
~~~~

<<dd_do:quietly>>
python:
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap=plt.cm.Spectral, edgecolor='none')
ax.view_init(30, 60)
plt.savefig("sandstone1.png")
end
<</dd_do>>


## 

![sandstone.png](./sandstone1.png)


## 动画 ([do-file](./stata/gif3d.do))

![sandstone.gif](./sandstone.gif)


# 网络数据的抓取

## 使用**pandas**获取表格

抓取[Nasdaq 100 stock tickers](https://en.wikipedia.org/wiki/NASDAQ-100)

~~~~
python:
import pandas as pd
data = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
df = data[2]
df = df.drop(df.index[0])
t = df.values.tolist()
end
~~~~

## 生成Stata dataset

~~~~
python:
from sfi import Data
Data.addObs(len(t))
stata: gen company = ""
stata: gen ticker = ""
Data.store(None, range(len(t)), t)
end
~~~~

## 

<<dd_do: quietly>>
use stata/nas100ticker.dta, clear
<</dd_do>>

~~~~
<<dd_do>>
list in 1/5, clean
<</dd_do>>
~~~~

## 使用**lxml**分解HTML

使用[Python script](./stata/nas1detail.py)获得Nasdaq 100股票具体信息, 
例如[ATVI](http://www.nasdaq.com/symbol/ATVI).


## 调用Python文件和输入参数

~~~~
use nas100ticker, clear
quietly describe
frame create detail
forvalues i = 1/`r(N)' {
	local a = ticker[`i']
	local b detail
	python script nas1detail.py, args(`a' `b')
	sleep 100
}
frame detail : save nasd100detail.dta, replace
~~~~

## 

<<dd_do: quietly>>
use stata/nasd100detail.dta, clear
keep if open_price != ""
<</dd_do>>

~~~~
<<dd_do>>
list ticker open_price open_date close_price close_date in 1/5, clean
<</dd_do>>
~~~~

# Support Vector Machine (SVM)

## **pysvm** ([ado file](./stata/pysvm.ado))

~~~~
program pysvm
	version 16
	syntax varlist, predict(name)
	gettoken label feature : varlist
	python: dosvm("`label'", "`feature'", "`predict'")
end
~~~~

## Python部分[ado file](./stata/pysvm.ado)

~~~~
python:
from sfi import Data
import numpy as np
from sklearn.svm import SVC

def dosvm(label, features, predict):
	X = np.array(Data.get(features))
	y = np.array(Data.get(label))

	svc_clf = SVC(gamma='auto')
	svc_clf.fit(X, y)

	y_pred = svc_clf.predict(X)

	Data.addVarByte(predict)
	Data.store(predict, None, y_pred)

end
~~~~

## 用**auto** dataset测试

<<dd_do: quietly>>
adopath + ./stata
<</dd_do>>

~~~~
<<dd_do>>
sysuse auto, clear
pysvm foreign mpg price, predict(for2)
<</dd_do>>
~~~~

## 比较结果

~~~~
<<dd_do>>
label values for2 origin
tabulate foreign for2, nokey
<</dd_do>>
~~~~

## 改进版

<<dd_do: quietly>>
sysuse auto, clear
set seed 12345
<</dd_do>>

~~~~
<<dd_do>>
pysvm2 foreign mpg price if runiform() <= 0.2
pysvm2predict for2
<</dd_do>>
~~~~

## 

~~~~
<<dd_do>>
label values for2 origin
tabulate foreign for2, nokey
<</dd_do>>
~~~~

## 训练程序([pysvm2.ado](./stata/pysvm2.ado))

~~~~
program pysvm2
	version 16
	syntax varlist(min=3) [if] [in]
	gettoken label features : varlist
	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as error "no observations"
		exit 2000
	}

	qui summarize `label' if `touse'
	if r(min) >= r(max) {
		di as error "outcome does not vary"
		exit 2000
	}

	quietly python: dosvm2("`label'", "`features'", "`touse'")
	di as text "note: training finished successfully"
end
~~~~

## Python部分[pysvm2.ado](./stata/pysvm2.ado)

~~~~
python:
import sys
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC
import __main__

def dosvm2(label, features, select):
	X = np.array(Data.get(features, selectvar=select))
	y = np.array(Data.get(label, selectvar=select))

	svc_clf = SVC(gamma='auto')
	svc_clf.fit(X, y)

	__main__.svc_clf = svc_clf
	Macro.setGlobal('e(svm_features)', features)
	return svc_clf
end
~~~~

## 预测程序([pysvm2predict.ado](./stata/pysvm2predict.ado))

~~~~
program pysvm2predict
	version 16
	syntax anything [if] [in]

	gettoken newvar rest : anything
	if "`rest'" != "" {
		exit 198
	}
	confirm new variable `newvar'
	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as text "zero observations"
		exit 2000
	}

	qui replace `touse' = _n if `touse' != 0 	
	python: dosvmpredict2("`newvar'", "`touse'")
end
~~~~

## Python部分[pysvm2predict.ado](./stata/pysvm2predict.ado)

~~~~
python:
import sys
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC
from __main__ import svc_clf

def dosvmpredict2(predict, select):
	features = select + " "+ Macro.getGlobal('e(svm_features)')
	X = np.array(Data.get(features, selectvar=select))

	y_pred = svc_clf.predict(X[:,1:])
	y1 = np.reshape(y_pred, (-1,1))

	y = np.concatenate((X, y1), axis=1)
	Data.addVarDouble(predict)
	dim = y.shape[0]
	j = y.shape[1]-1
	for i in range(dim):
		Data.storeAt(predict, y[i,0]-1, y[i,j])

end
~~~~

# ado程序间传递Python实例

##

In [pysvm2.ado](./stata/pysvm2.ado) ado code:

~~~~
...
import __main__
...
__main__.svc_clf = svc_clf
...
~~~~

##
To access **svc_clf** in Python routines in ado files:

~~~~
...
from __main__ import svc_clf
...
~~~~

# 工具

- **python query**
- **python describe**
- **python set exec**
- **python search**



# 词语分析

##

![中文词图](./stata/words2.png)

## **jieba**[中文分词](./stata/words2.do)

~~~~
url = "http://www.uone-tech.cn/news/stata16.html"    
html = requests.get(url) 
html.encoding='utf-8' 
text = BeautifulSoup(html.text).get_text() 

import jieba       
words = jieba.lcut(text)        

~~~~

## 

![words.png](./stata/words.png)

## [英文词频](./stata/words.do)

~~~~
url = "https://www.stata.com/new-in-stata/python-integration/"    
html = requests.get(url)  
text = BeautifulSoup(html.text).get_text() 

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=75, 
	max_words=100, 
	background_color="white").generate(text)
~~~~

# 谢谢!

# Post-credits...

- [sfi details and examples][sfi]
- [Stata Python documentation][P python]
- [Stata Python integration](https://www.stata.com/new-in-stata/python-integration/)
- The talk is made with [Stata markdown](https://www.stata.com/features/overview/markdown/) and [dynpandoc](https://ideas.repec.org/c/boc/bocode/s458455.html)
- [wordcloud do-file](./stata/words.do)


[hpeng]: hpeng@stata.com
[sfi]: https://www.stata.com/python/api16/
[P python]:https://www.stata.com/manuals/ppython.pdf
