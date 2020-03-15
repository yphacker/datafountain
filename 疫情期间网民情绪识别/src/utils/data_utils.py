# coding=utf-8
# author=yphacker

import re


def clean_name(x):
    # x = x.replace()
    x = re.sub(r'//@.*?:', ',', x)
    x = re.sub(r',+', ',', x)
    if x[0] == ',':
        x = x[1:]
    return x


def clean_huati(x):
    x = re.sub(r'#.*?#', '', x)
    return x


if __name__ == '__main__':
    print(clean_name('日常反腐#南京灯光秀#//@河北王敏杰A://@游泳的鱼_37952:邓煌灯光秀炫腐乌龟莫忧愁，大伞默'))
    print(clean_name('//@路过全世界的白小狸:那孝感人民…一一谢过~~'))
    print(clean_huati(clean_name('日常反腐#南京灯光秀#//@河北王敏杰A://@游泳的鱼_37952:邓煌灯光秀炫腐乌龟莫忧愁，大伞默')))
