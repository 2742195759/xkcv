import pandas as pd
dataset_path = '../data/dataset.txt'

ds = pd.read_csv(dataset_path, header=None, names=['uid', 'iid', 'review', 'quality'])
print (ds[0:10])
# print (ds['review'])

# start process 
user_num = len(ds['uid'].unique())
item_num = len(ds['iid'].unique())
interact_num = len(ds)
interact_rate = interact_num * 1.0 / (user_num * item_num)

print ('================================基本信息===============================')

print ('user_num', 'item_num', 'interact_num', 'interact_rate')
print (user_num, item_num, interact_num, interact_rate)

user_len = ds.groupby(by=['uid']).apply(len)
assert (len(user_len[user_len>=1]) == user_num)
print ('>5 users', len(user_len[user_len>=5]))
print ('>10 users', len(user_len[user_len>=10]))
print ('>20 users', len(user_len[user_len>=20]))

print ('===============================同图不同User Caption=====================')
image_len = ds.groupby(by=['iid']).apply(len).sort_values(ascending=False)
print (image_len[0:10])
idx = 0 # 前idx个图片的所有评论
(ds[ds['iid'] == image_len.index[idx]]).to_csv('../data/case_imageid.csv')
print ()
print ('Concrete Reviews in ../data/case_imageid.csv')

print ('=============================验证，没有一个对同一个图片两个评论========')
tmp = ds.groupby(by=['uid','iid']).apply(len)
print ('这么多个', len(tmp[tmp>1]), '是重复评论')
print ('示例:')
print (ds[ds.duplicated(subset=['uid','iid'], keep=False)][0:10].sort_values(by=['uid','iid']).reset_index(drop=True))
