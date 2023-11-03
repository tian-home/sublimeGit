# -*- coding:utf-8 -*-
anchors = [{'title': '蒜香莲藕虾', 'mat': '用料：莲藕、虾、蒜、葱、酱油、盐、白糖、水、植物油', 'collect': 28}, {'title': '小炒金钱肚', 'mat': '用料：金钱肚、青椒、胡萝卜、洋葱、小米辣、生姜、大蒜头、油、生抽、白糖、盐、黄酒', 'collect': 98}, {'title': '手抓牛肉', 'mat': '用料：牛肉、胡萝卜、豆角、香菇、蒜、姜、蒜苗', 'collect': 19}, {'title': '茯苓姬松茸排骨汤', 'mat': '用料：姬松茸、莲子、百合、茯苓、生姜、排骨、盐、纯净水、料酒', 'collect': 92}, {'title': '凉拌娃娃菜', 'mat': '用料：娃娃菜、小米椒、蒜、醋、生抽、香油', 'collect': 95}]

# anchors = sorted(anchors,key=collect)
anchors.sort(key=lambda x:x["collect"], reverse=True)

for s in anchors:
    print(s)