# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:54:19 2022

@author: V Borges Rodrigues
"""
##############################################################################

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trec
import pprint as pp
import pickle

Qrels = "qrels-clinical_trials.txt"
Queries = "topics-2014_2015-summary.topics"


with open(Queries, 'r') as queries_reader:
    txt = queries_reader.read()

root = ET.fromstring(txt)

cases = {}
cases_age = {}
cases_genre = {}
for query in root.iter('TOP'):
    q_num = query.find('NUM').text
    q_title = query.find('TITLE').text
    cases[q_num] = q_title
    #cases_age[q_num] = query.find('AGE').text
    #cases_genre[q_num] = query.find('GENDER').text

eval = trec.TrecEvaluation(cases, Qrels)

pickle.dump(cases, open("cases.bin", "wb"))
#pickle.dump(cases_age, open("cases_age.bin", "wb"))
#pickle.dump(cases_genre, open("cases_genre.bin", "wb"))

##############################################################################


# processar documentos
import xml.etree.ElementTree as ET
import tarfile

tar = tarfile.open("clinicaltrials.gov-16_dec_2015.tgz", "r:gz")
ids = []
brief_titles = []
detailed_descriptions = []
brief_summaries = []
criterias = []
genders = []
minimum_ages = []
maximum_ages = []

namelist = tar.getnames()

for tarinfo in tar:
    if tarinfo.size > 500:
        txt = tar.extractfile(tarinfo).read().decode("utf-8", "strict")
        root = ET.fromstring(txt)
        judged = False
        id=''
        bt=''
        dt=''
        bs=''
        cs=''
        ge=''
        ma=''
        mi=''

        for doc_id in root.iter('nct_id'):
            if doc_id.text in eval.judged_docs:
                judged = True
                id=doc_id.text.strip()
                break
                
        if judged is False:
            continue

        brief_title = root.find('brief_title').text.strip()
        bt=brief_title

        for dd in root.iter('detailed_description'):
            for child in dd:
                dd=child.text.strip()
        if dd == '':
            dd=brief_title
              
        for bs in root.iter('brief_summary'):
            for child in bs:
                bs=child.text.strip()
        if bs == '':
            bs=brief_title
                
        for c in root.iter('criteria'):
            for child in c:
                cs=child.text.strip()
        if cs=='':
            cs= brief_title

        for gender in root.iter('gender'):
            ge=gender.text.strip()
        if ge=='' or ge=='N/A':
            ge='Both'

        for minimum_age in root.iter('minimum_age'):
            mi=minimum_age.text.strip()
        if mi=='' or mi=='N/A':
            mi='0 years'

        for maximum_age in root.iter('maximum_age'):
            ma=maximum_age.text.strip()
        if ma=='' or ma=='N/A':
            ma='100 years'

       
        if id!='' and bt!='' and dd!='' and bs!='' and cs!='' and ge!='' and mi!='' and ma!='':
            ids.append(id)
            brief_titles.append(bt)
            detailed_descriptions.append(dd)
            brief_summaries.append(bs)
            criterias.append(cs)
            genders.append(ge)
            minimum_ages.append(mi)
            maximum_ages.append(ma)
        else:
            print('failed')
            print(id,bt,dd,bs,cs)
            continue

        if len(ids) != len(brief_titles) or len(detailed_descriptions) != len(brief_summaries) or len(brief_summaries) != len(criterias):
            print("error")
            print(brief_title)
            print(detailed_descriptions[0].text.strip())
            print(brief_summaries[0].text.strip())
            print(criterias[0].text.strip())
            break
tar.close()

pickle.dump(ids, open("ids.bin", "wb"))
pickle.dump(brief_titles, open("brief_title.bin", "wb"))
pickle.dump(detailed_descriptions, open("detailed_description.bin", "wb"))
pickle.dump(brief_summaries, open("brief_summary.bin", "wb"))
pickle.dump(criterias, open("criteria.bin", "wb"))
pickle.dump(genders, open("gender.bin", "wb"))
pickle.dump(minimum_ages, open("minimum_age.bin", "wb"))
pickle.dump(maximum_ages, open("maximum_age.bin", "wb"))


##############################################################################


### initiate class

import abc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
from collections import Counter
import re

def flatten_arr(l):
    return [item for sublist in l for item in sublist]

#Definir Classe Abstrata
class RetrievalModel:
    def __init__(self,queries_idx,labels,docs_parts):
        self.idx_cases=queries_idx
        self.labels=labels
        self.features_model=[]
        self.doc_parts=docs_parts
        self.train=None
        self.test=None
        self.coefs=[]
        self.training_performance=[]
    
    def create_features(self):
        counter=0
        for corpus in self.doc_parts:
            model1=VSM("bigram","english")
            model1.search(self.idx_cases,corpus)
            if counter==0:
                #self.features_model=self.fmodel1.rename(columns={'score':'score'+str(counter)})
                self.features_model = model1.get_features().rename(columns={'score':'score'+str(counter)})
                counter+=1
            else:
                #features_model.join(model1.get_features,on=['A', '_id'])
                #features_model.join(other.set_index(['A','_id']), on=['A', '_id'], lsuffix='_caller', rsuffix='_other')
                to_merge=model1.get_features().rename(columns={'score':'score'+str(counter)})
                self.features_model=pd.merge(self.features_model,to_merge,on=['A', '_id'],how='left' )
                counter+=1
            model2=LMJM("unigram","english",)
            model2.search(self.idx_cases,corpus,0.4)
            if counter==0:
                self.features_model=model2.get_features()#.rename(columns={'score_lmjm':'score_lmjm'+str(counter)})
                counter+=1
            else:
                to_merge=model2.get_features()#.rename(columns={'score_lmjm':'score_lmjm'+str(counter)})
                self.features_model=pd.merge(self.features_model,to_merge,on=['A', '_id'],how='left' )
                counter+=1
             
    def process(self): 
        # preprocess first
        self.features_model['A']= self.features_model['A'].astype(int)
        joined=pd.merge(self.features_model,self.labels,on=['A', '_id'],how='left')
        cols_to_scale = joined.columns[2:-1]
        #create and fit scaler
        scaler = StandardScaler()
        scaler.fit(joined[cols_to_scale])
        #scale selected data
        joined[cols_to_scale] = scaler.transform(joined[cols_to_scale])
        # take out row columns witn NAN label and make it binary
        joined.loc[joined['label'] == 2, 'label'] = 1
        joined['A']=joined['A'].astype(str)
        joined.columns = ['A', '_id', 'vsm1','lmjm1','vsm2','lmjm2','vsm3','lmjm3','vsm4','lmjm4','label']
        self.features_model = joined
        print(self.features_model.isna().sum())
        # fill nan with the most similar attribute
        self.features_model.lmjm3.fillna(self.features_model.lmjm2, inplace=True)
        train, test = train_test_split(self.idx_cases, test_size=0.2)
        self.train = self.features_model.loc[self.features_model['A'].isin(train)]
        self.test = self.features_model.loc[self.features_model['A'].isin(test)].reset_index(drop=True)


    def fit(self,w='balanced'):
        #treinar com cross validation K=5
        self.train=self.train.reset_index(drop=True)
        # drop non label rows
        train_test=self.train.iloc[:,:-1]
        self.train.dropna(subset=['label'], how='all', inplace=True)
        X = self.train.iloc[:, :].reset_index(drop=True)
        c_l = np.arange(0.1,12,0.5)
        performance = []
        k = 5
        kf = KFold(n_splits=k,shuffle=True)
        ids=X.A.unique()
        for c in c_l:
            p10_avg=[]
            for train_idx,test_idx in kf.split(ids):
                ids_train = [ids[i]  for i in train_idx]
                ids_test = [ids[i] for i in test_idx]
                Xtrain = X.loc[X['A'].isin(ids_train)]
                Xtest = X.loc[X['A'].isin(ids_test)]
                X_train, Y_train = Xtrain.iloc[:,:-1], Xtrain.iloc[:,-1]
                X_test, Y_test = Xtest.iloc[:,:-1], Xtest.iloc[:,-1]
                clf = LogisticRegression(random_state=0,C=c,class_weight=w).fit(X_train.iloc[:,2:], Y_train)
                coefs_train = clf.coef_[0]
                X_test.iloc[:, 2:] * coefs_train
                X_test['score'] = X_test.iloc[:, 2:].sum(axis=1)
                p10_t= 0
                counter = 0
                for caseid in X_test.A.unique():
                    res = X_test.loc[X_test["A"] == caseid, ['_id', 'score']]
                    res_ord = res.sort_values(by=['score'], ascending=False)
                    p10, recall, ap, ndcg5, mrrv = eval.eval(res_ord, caseid)
                    p10_t += p10
                    counter += 1
                p10_avg.append(p10_t/counter)
            performance.append(sum(p10_avg)/len(p10_avg))
        performance = np.array(performance)
        print(performance)
        self.training_performance=performance
        betterc = c_l[np.where(performance==np.amin(performance))]
        clf= LogisticRegression(random_state=0,C=betterc[0],class_weight='balanced').fit(X.iloc[:,2:-1],X.iloc[:,-1])
        self.coefs = clf.coef_[0]

    def get_coefs(self):
        return self.coefs

    def get_featuresfortest(self):
        return self.test

    def get_performance(self):
        return self.training_performance

        
class VSM:  # subclasse do nosso retrievalmodel modelo VSM
    def __init__(self,tipo,stopw='english'):
        self.tipo=tipo
        self.stopw=stopw
        self.ranked_docs_per_query={}
        self.score_per_query={}
        self.total_score_model=[]
        self.features=None
    def search(self, caseids, docs):  # método search que calcula a distancia entre o querie e os diferentes documentos
        '''este método recebe 2 argumentos (paciente id e docs) de seguida vectoriza os documentos e depois mede a 
        distancia entre o query e os documentos. retorna os scores para cada documento'''
        n=1
        if self.tipo=="unigram":
            pass
        elif self.tipo=="bigram":
            n=2
        else:
            raise ValueError('unigram or bigram')
        index = TfidfVectorizer(ngram_range=(1, n), stop_words=self.stopw)
        index.fit(docs)
        X = index.transform(docs)
        for caseid in caseids:
            query = cases[caseid]
            query_tfidf = index.transform([query])
            doc_scores = 1-pairwise_distances(X, query_tfidf, metric='cosine')
            self.ranked_docs_per_query[caseid]=doc_scores.tolist()
        control=0
        for caseid,scores in self.ranked_docs_per_query.items():
            results_vsm = pd.DataFrame(list(zip(ids, flatten_arr(scores))), columns = ['_id', 'score'])
            results_vsm.insert(loc=0, column='A', value=caseid)       
            if control==0:
                self.features=results_vsm
            else:
                self.features=pd.concat([self.features,results_vsm], ignore_index = True)
                self.features.reset_index()
            control+= 1
            
            

    def evaluate(self,ids):
        p10_t, recall_t, ap_t, ndcg_t, mrr_t=0,0,0,0,0
        counter=0
        for caseid,scores in self.ranked_docs_per_query.items():
            results_vsm = pd.DataFrame(list(zip(ids, scores)), columns = ['_id', 'score'])
            results_ord_vsm = results_vsm.sort_values(by=['score'], ascending = False)
    
            p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm = eval.eval(results_ord_vsm, caseid)
            p10_t+=p10vsm
            recall_t+=recallvsm
            ap_t+= apvsm
            ndcg_t+=ndcg5vsm
            mrr_t+=mrrvsm
            self.score_per_query[caseid]= [p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm]
            counter+=1
        self.total_score_model=[p10_t/counter, recall_t/counter, ap_t/counter, ndcg_t/counter, mrr_t/counter]
        
    def get_score(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.total_score_model
    def get_queryscores(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.score_per_query
    def get_rankeddocs(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.ranked_docs_per_query 
    def get_features(self):
        return self.features
            
            


##### LMJM CLASS ...

class LMJM: # subclasse do nosso retrievalmodel. modelo LM Jellineck Mercier Smoothing
    def __init__(self,tipo,stopw='english'):
        self.tipo=tipo
        self.stopw=stopw
        self.ranked_docs_per_query={}
        self.score_per_query={}
        self.total_score_model=[]
        self.features=None
        
    def search(self, caseids, docs,lmbd_insert):
        '''Este métod tem como argumentos o case id e os documentos. vectoriza os documentos e de seguida calcula
        as diferentes probabilidades para os queries'''
        n=1
        if self.tipo=="unigram":
            pass
        elif self.tipo=="bigram":
            n=2
        else:
            raise ValueError('unigram or bigram')
        index = CountVectorizer(ngram_range=(1, n), analyzer='word',stop_words=self.stopw)
        corpus_cv = index.fit(docs).transform(docs)
        lmbd = lmbd_insert
        prob_word_docs = corpus_cv/np.sum(corpus_cv, axis=1)  # p(t|md)
        prob_word_corpus = np.sum(corpus_cv, axis=0) / np.sum(corpus_cv)  # p(t|mc)
        log_mixture = np.log(lmbd*prob_word_docs + (1-lmbd)*prob_word_corpus)
        for caseid in caseids:
            query = cases[caseid]
            query_cv = index.transform([query])
            total = log_mixture*query_cv.T
            self.ranked_docs_per_query[caseid]=total.flatten().tolist()
        control=0
        for caseid,scores in self.ranked_docs_per_query.items():
            results_LMJM = pd.DataFrame(list(zip(ids, scores[0])), columns = ['_id', 'score_lmj'])
            results_LMJM.insert(loc=0, column='A', value=caseid)
            if control==0:
                self.features=results_LMJM
            else:
                self.features=pd.concat([self.features,results_LMJM], ignore_index = True)
                self.features.reset_index()
            control+=1
            
    def evaluate(self,ids):
        p10_t, recall_t, ap_t, ndcg_t, mrr_t=0,0,0,0,0
        counter=0
        for caseid,scores in self.ranked_docs_per_query.items():
            results_LMJM = pd.DataFrame(list(zip(ids, scores[0])), columns = ['_id', 'score_lmj'])
            results_ord_LMJM = results_LMJM.sort_values(by=['score_lmj'], ascending = False)

            p10, recall, ap, ndcg5, mrr = eval.eval(results_ord_LMJM, caseid)
            p10_t+=p10
            recall_t+=recall
            ap_t+= ap
            ndcg_t+=ndcg5
            mrr_t+=mrr
            self.score_per_query[caseid]= [p10, recall, ap, ndcg5, mrr]
            counter+=1
            #if (np.shape(recall_11point) != (0,)):
                #avg_precision_11point_t = avg_precision_11point + precision_11point
        self.total_score_model=[p10_t/counter, recall_t/counter, ap_t/counter, ndcg_t/counter, mrr_t/counter]
    def get_score(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.total_score_model
    def get_queryscores(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.score_per_query
    def get_rankeddocs(self):
        "este metodo devolve uma lista com o score total do modelo"
        return self.ranked_docs_per_query 
    def get_features(self):
        return self.features
    
def evaluation(coef,features,drop=False,model='score'):
    results=features
    if drop ==True:
        results.dropna(subset=['label'], how='all', inplace=True)
    # print(results)
    if coef!='model':
        results.iloc[:, 2:-1] * coef
    # print(results)
    results['score']=results.iloc[:, 2:-1].sum(axis=1)
    score_by_query={}
    precision_11point_t, recall_11point_t, total_relv_ret_t=[],[],[]
    p10_t, recall_t, ap_t, ndcg_t, mrr_t = 0, 0, 0, 0, 0
    score_model=0
    counter=0
    for caseid in results.A.unique():
        res = results.loc[results["A"] == caseid, ['_id', model]]
        res_ord = res.sort_values(by=[model], ascending=False)
        p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm = eval.eval(res_ord, caseid)
        precision_11point, recall_11point, total_relv_ret = eval.evalPR(res_ord, caseid)  
        p10_t += p10vsm
        recall_t += recallvsm
        ap_t += apvsm
        ndcg_t += ndcg5vsm
        mrr_t += mrrvsm
        score_by_query[caseid] = [p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm]
        counter += 1
    if (np.shape(recall_11point) != (0,)):
        recall_11point_t.append(recall_11point) 
    score_model = [p10_t / counter, recall_t / counter, ap_t / counter, ndcg_t / counter, mrr_t / counter]
    return score_model,score_by_query,precision_11point_tv,recall_11point_tv,counter

##############################################################################

# split into train and test
idx_cases=list(cases.keys())

# pre-processing (keep only the lines that have labels) comparing features keys with qrles.txt file keys
data_qrl = pd.read_csv('qrels-clinical_trials.txt', delimiter=r"\s+", header=None,names=["A", "b", "_id", "label"])
data_qrl=data_qrl.drop('b',axis=1)
#features_model['A']=features_model['A'].astype(int)
doc_parts=(brief_titles, brief_summaries,detailed_descriptions, criterias)


########## implement LETOR
model_ir= RetrievalModel(idx_cases,data_qrl,doc_parts)
model_ir.create_features()
model_ir.process()
model_ir.fit()


# save performances,coeficients and features for model1 - logreg balanced
model_ir.get_featuresfortest().to_pickle("features_for_test_1.pkl")
with open('coeficientes1.pkl', 'wb') as f:
    pickle.dump(model_ir.get_coefs(), f)
with open('training_performance1.pkl', 'wb') as f:
    pickle.dump(model_ir.get_performance(), f)
    
# save performances,coeficients and features for model2 - logreg non balanced
model_ir.fit(w=None)
model_ir.get_featuresfortest().to_pickle("features_for_test_2.pkl")
with open('coeficientes2.pkl', 'wb') as f:
    pickle.dump(model_ir.get_coefs(), f)
with open('training_performance2.pkl', 'wb') as f:
    pickle.dump(model_ir.get_performance(), f)

# save performances,coeficients and features for model3 - logreg balanced mannualy
model_ir.fit(w={0:0.8, 1:0.2})
model_ir.get_featuresfortest().to_pickle("features_for_test_3.pkl")
with open('coeficientes3.pkl', 'wb') as f:
    pickle.dump(model_ir.get_coefs(), f)
with open('training_performance3.pkl', 'wb') as f:
    pickle.dump(model_ir.get_performance(), f)



# evaluate different models
def features_4test(i):
    features = pd.read_pickle(f'features_for_test_{i}.pkl')
    with open(f'coeficientes{i}.pkl', 'rb') as f:
        coefs = pickle.load(f)
    return features,coefs

# evaluate model1 (balanced)
features1,coefs1=features_4test(1)
score_model1,score_by_query,precision_t,recall_t,counter=evaluation(coefs1,features1)
score_model1

# evaluate model2 (unbalanced)
features2,coefs2=features_4test(2)
score_model2,score_by_query2,precision_t2,recall_t2,counter=evaluation(coefs2,features2)
score_model2
# evaluate model3 (mannualy)
features3,coefs3=features_4test(3)
score_model3,score_by_query3,precision_t3,recall_t3,counter=evaluation(coefs3,features3)
score_model3

# evaluate models in all documents parts qith vsm and lmjm
models=['vsm1','lmjm1','vsm2','lmjm2','vsm3','lmjm3','vsm4','lmjm4']

for i in models:
    score_model,score_by_query,precision_t,recall_t,counter = evaluation('model',features1,False,i)
    print(score_model)
    with open(f'score_{i}.pickle', 'wb') as file:
        pickle.dump(score_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'score_by_query_{i}.pickle', 'wb') as file:
        pickle.dump(score_by_query, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'precision_t_{i}.pickle', 'wb') as file:
        pickle.dump(precision_t, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'recall_t_{i}.pickle', 'wb') as file:
        pickle.dump(recall_t, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'counter_{i}.pickle', 'wb') as file:
        pickle.dump(counter, file, protocol=pickle.HIGHEST_PROTOCOL)




