# 口罩佩戴检测大赛
[竞赛链接](https://www.datafountain.cn/competitions/422)
## 数据下载
[数据链接](https://www.datafountain.cn/competitions/422/datasets)
## 评估标准

## score
chinese_roberta_wwm_ext  
content  
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert|-1|-1||

content + comment_2c
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert|-1|-1||

content + comment_all
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert|-1|-1||

## script
nohup python main.py -task=0 -m='bert' -b=50 -e=8 -mode=2 > nohup/bert_task0.out 2>&1 &
nohup python main.py -task=1 -m='bert' -b=35 -e=8 -mode=2 > nohup/bert_task1.out 2>&1 &
nohup python main.py -task=2 -m='bert' -b=35 -e=8 -mode=2 > nohup/bert_task2.out 2>&1 &

python main.py -o=predict -task=3

## note
