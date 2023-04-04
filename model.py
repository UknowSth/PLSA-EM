from collections import defaultdict
import jieba
import numpy as np
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

class PLSA:
    def __init__(self,path=None,iteration=20,k=10,topk=5):
        '''
            iteration : 默认EM算法迭代次数为20
            k : 主题数量, 默认为10
        '''
        self.path = path # 可以传入待分词的文本文件，也可以传入文本路径构成的列表
        self.iteration = iteration
        self.K = k
        self.topk = topk
        self.words = []
        self.n_dw = {} # n(d,w) M,N

    def parse_int_keys(self,pairs):
        result = {}
        for key, value in pairs:
            try:
                # Attempt to parse key as integer
                key = int(key)
            except ValueError:
                pass
            result[key] = value
        return result

    def getWords(self,text):
        words = jieba.lcut(text)
        Temp = defaultdict(int)
        for word in words:
            if word not in self.stop_words and len(word) > 1 and word.isdigit() is not True: # 排除停用词和单字词
                Temp[word] += 1
                if word not in self.words:
                    self.words.append(word)
        return Temp

    def load_para(self,path):
        '''
            如果使用加载的数据只能调用printTopK和showgraph函数
            为了可以接着train!
        '''
        with open(path, 'r',encoding='utf-8') as f:
            para = json.load(f,object_pairs_hook=self.parse_int_keys)
        self.p_wz = np.array(para['wz'])
        self.p_zd = np.array(para['zd'])
        self.n_dw = para['dw']
        self.words = para['words']
        print('shape of p_wz',self.p_wz.shape)
        print('shape of p_zd',self.p_zd.shape)
        self.K = self.p_wz.shape[1]
        self.M = self.p_zd.shape[1]
        self.N = self.p_wz.shape[0]
        self.dict_words = {key:value for value,key in enumerate(self.words)}
    
    def save_para(self,path):
        para = {'wz':self.p_wz.tolist(),'zd':self.p_zd.tolist(),'words':self.words,'dw':self.n_dw}
        with open(path,'w',encoding='utf-8')as o:
            json.dump(para,o)
        print("参数已存储完毕")

    def load(self,mode='common',iters=50000):
        # 为了区分不同的数据集 采用不同的操作
        print("正在加载数据...")
        # 加载停用词
        with open('./stop_word.txt','r',encoding='utf-8')as w:
            self.stop_words = w.read().split('\n')
        if isinstance(self.path,list):
            self.M = len(self.path)
            print('共计{}篇文档.'.format(len(self.path)))
            if mode != 'common': # 继续之前读入的数据加载
                with open('temp.json','r',encoding='utf-8') as f:
                    para = json.load(f,object_pairs_hook=self.parse_int_keys)
                self.words = para['words']
                self.n_dw = para['dw']
                count = para['count']
            else:
                count = 0
            for i in range(count,min(count+iters,len(self.path))):
                p = self.path[i]
                with open(p,'r',encoding='utf-8')as f:
                    self.n_dw[i] = self.getWords(f.read())
                if i%1000 == 0:
                    print("已经扫描了{}数据了".format(i))
            if(i == len(self.path)-1):
                print("全部数据已扫描完成")
            else:
                para = {'words':self.words,'dw':self.n_dw,'count':i+1}
                with open('temp.json','w',encoding='utf-8')as f:
                    json.dump(para,f)
                print("加载过程的中间参数已经存储完毕.count={}".format(i+1))

        else:
            with open(self.path,'r',encoding='utf-8') as f:
                text = f.read().split('\n')
            text.remove('')
            self.M = len(text)
            print("共计{}篇文档.".format(len(text)))

            # 计算n(d,w) M,N
            for i,doc in enumerate(text):
                self.n_dw[i] = self.getWords(doc)

        self.dict_words = {key:value for value,key in enumerate(self.words)} # 构建词到index的映射
        self.N = len(self.words)
        print("共计{}个词.".format(self.N))

    def initialize(self):
        # 初始化参数 均匀分布
        self.p_wz = np.random.rand(self.N,self.K)
        self.p_wz = self.p_wz/self.p_wz.sum(axis=0,keepdims=True) # N,K
        self.p_zd = np.random.rand(self.K,self.M)
        self.p_zd = self.p_zd/self.p_zd.sum(axis=0,keepdims=True) # K,M

        
    def getN(self,j): # 是否要做平滑?
        # 把字典形式的n(d,w)变成向量
        res = np.zeros((self.N,1))
        for key,value in self.n_dw[j].items():
            res[self.dict_words[key]] = value
        return res

    def update(self):
        # 首先计算出n_d便于后续计算
        self.n_d = np.zeros((1,self.M))
        for key,value in self.n_dw.items():
            self.n_d[0,key] = sum(value.values())

        for i in range(self.iteration):
            # updata p(w|z) and p(z|d)
            new_wz = np.zeros((self.N,self.K))
            new_zd = np.zeros((self.K,self.M))
            for j in range(self.M):
                temp = self.p_wz * self.p_zd[:,j].T
                temp = temp/temp.sum(axis=1,keepdims=True) # normalize N*K 对K做
                new_wz += self.getN(j)*temp
                new_zd[:,j] = (self.getN(j).T @ temp).T.reshape((self.K,))
            self.p_zd = new_zd/self.n_d
            self.p_wz = new_wz/new_wz.sum(axis=0,keepdims=True) # normalize 对N做
            del new_wz
            del new_zd
            if i % 5 == 0:
                self.printTopK(self.topk)
                print('-----------------------------------------------')

    def top_k_indexes(self,arr, k):
        idx = np.argpartition(arr, -k)[-k:]
        return idx[np.argsort(-arr[idx])]

    def printTopK(self,k):
        '''
            k: 每个主题下打印前k个词
            根据p(w|z)的结果进行打印
        '''
        temp = [self.top_k_indexes(self.p_wz[:, i], k) for i in range(self.p_wz.shape[1])] # 求topk对应的下标
        for i,t in enumerate(temp):
            print("以下是第{}类中的topk word:".format(i))
            print([self.words[u] for u in t])
        
    def show_graph(self,k):
        '''
            根据前k个词绘制词云
        '''
        data = []
        for j in range(self.K):
            Idx = self.top_k_indexes(self.p_wz[:,j],k)
            D = {self.words[i]:self.p_wz[i,j] for i in Idx}
            data.append(D)
        fig, axes = plt.subplots(nrows=int((self.K+1)/2), ncols=2, figsize=(20,40))
        # font_path='./simsun.ttc'
        font_path = './msyh.ttc'
        mask = np.array(plt.imread('./graph.webp'))
        for i in range(self.K):
            row, col = divmod(i, 2)
            words = data[i] # min_font_size=25, max_font_size=50
            wordcloud = WordCloud(mask=mask,font_path=font_path,width=300, height=200, background_color='white').generate_from_frequencies(words)
            axes[row,col].imshow(wordcloud, interpolation='bilinear')
            axes[row,col].axis('off')
            axes[row,col].set_title(f"Word Cloud {i+1}")
        plt.tight_layout()
        plt.show()
        return