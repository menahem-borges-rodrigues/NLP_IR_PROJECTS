# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file from Menahem_Borges_Rodrigues_linux02
"""
# import libraries
import random
import pickle
from bertviz import model_view, head_view
import transformers
import numpy as np
import pprint
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from transformers import BertTokenizer, BertModel,AutoTokenizer,AutoModel,AutoConfig
import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns; sns.set_theme()
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
import trec
##############################################################################


def get_data(lista):
    '''this function receive a list of pickle file names and returns
    the data (cases,doc_ids,doc_brief_summaries)'''
    with open(lista[0], 'rb') as f:
        cases= pickle.load(f)
    with open(lista[1], 'rb') as f:
        ids= pickle.load(f)
    with open(lista[2], 'rb') as f:
        brief_summary= pickle.load(f)
    with open(lista[3], 'rb') as f:
        det_description= pickle.load(f)
    return cases,ids,brief_summary,det_description

## TOKENIZER
def sentences_encoder (sentence_1,sentence_2):
    '''esta função recebe duas frases e retorna os tokens, bem como os embbedings''' 
    inputs = tokenizer.encode_plus(sentence_1,sentence_2, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)
    inputs_list = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs_list[0].tolist())
    ## MODEL INFERENCE
    with torch.no_grad():
        outputs = model(**inputs)
    return inputs,tokens,outputs

# EXTRACT 10 TOKEN and EMBEDDINGS 1st and lst layer
def extractor_tokens_embeddings(tokens,size,outputs):
    '''esta função recebe os tokens e os seus outputs e retorna 
    10 tokens aleatórios bem como os embbedings da primeira e ultima layer.'''
    tokens_20=[]
    z=[i for i in range(0,len(tokens))]

    for i,j in zip(tokens,z):
        if len(i) >= 4:
            tokens_20.append((i,j))
      
    # choose 10 random tokens and get their index
    tokens_rand=random.sample(tokens_20, size)

    tokens_10_pos=[j for i,j in tokens_rand]
    tokens_10=[i for i,j in tokens_rand]
    tokens_10_pos.insert(0,0)
    tokens_10.insert(0,'[CLS]')
    
    # get the embedings of the first hidden layer and last layer
    first_emb=[]
    last_emb=[]
    output_embeddings = outputs['last_hidden_state']
    for i in tokens_10_pos:    
        last_emb.append(output_embeddings[0][i].numpy())
    output_embeddings_hidden = outputs['hidden_states']
    for i in tokens_10_pos:    
        first_emb.append(output_embeddings_hidden[0][0][i].numpy())
        
    first_emb ,last_emb=np.array(first_emb),np.array(last_emb)
    return tokens_10,first_emb,last_emb
    
def extractor_tokens_query_docs(tokens,outputs):
    '''esta função recebe tokens e outputs e retorna 
    uma lista de tokens e as suas embbedings, sendo essa lista dividida em CLS + 7 tokens respeitantes
    à query + 10 tokens referentes à detailled description divididos pelo SEP.
    Os tokens são escolhidos de forma totalmente aleatótia.'''
    tokens_control=[]
    tokens_heat=['[CLS]']
    tokens_heat_pos=[0]
    z=[i for i in range(0,len(tokens))]
    for i,j in zip(tokens,z):
        if len(i) > 4:
            tokens_control.append((i,j))

    sep_pos=0
    sep_control=0
    for i,j in tokens_control:
        if i == '[SEP]':
            sep_pos=j
            break
        sep_control+=1
    # choose random 10 tokens
    if sep_pos <= 10:
        size = sep_pos-1
    else:
        size = 10
    tokens_rand_q=random.sample(tokens_control[1:sep_control], size) 
    tokens_rand_doc=random.sample(tokens_control[sep_control:-1], 10)        
    for i,j in tokens_rand_q:
        tokens_heat.append(i)
        tokens_heat_pos.append(j)
    counter=0
    for i,j in tokens_rand_doc:
        if counter==0:
            tokens_heat.append('[SEP]')
            tokens_heat_pos.append(sep_pos)
            tokens_heat.append(i)
            tokens_heat_pos.append(j)
            counter+=1
        else:
            tokens_heat.append(i)
            tokens_heat_pos.append(j)
    last_emb_heat=[]
    first_emb_heat=[]
    output_embeddings = outputs['last_hidden_state']
    output_embeddings_hidden = outputs['hidden_states']
    for i in tokens_heat_pos:    
         last_emb_heat.append(output_embeddings[0][i].numpy())

    for i in tokens_heat_pos:    
         first_emb_heat.append(output_embeddings_hidden[0][0][i].numpy())
         
    first_emb_heat ,last_emb_heat=np.array(first_emb_heat),np.array(last_emb_heat)
    return tokens_heat,first_emb_heat,last_emb_heat  
            
        
# GET 2 FEATURES Of OUR EMBEDDINGS (USING TSNE)

def compute_tsne_plot(labels,first_emb,lst_emb):
    ''' esta função recebe os tokens 'labls' e os respectivos embbedings da primeira e ultima 
    layers e retorna um plot, reduzindo dimensionalmente o input de entrada de tamanho [2,748]
    para [2,2] recorrendo ao TSNE'''
    X_1 = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(first_emb)
    X_f = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(lst_emb)

    #plot figure with embbedings TSNE 
    plt.scatter(X_1[:,0],X_1[:,1], marker='o')
    plt.scatter(X_f[:,0],X_f[:,1], marker='o',c='red')
    for label, x, y in zip(labels, X_1[:,0], X_1[:,1]):
        plt.annotate(
            label,
            xy=(x, y))
    for label, x, y in zip(labels, X_f[:,0], X_f[:,1]):
        plt.annotate(
            label,
            xy=(x, y))    
    plt.show()

def extract_cls(query_pairs, embeddings, batch_size=16):
    '''esta função recebe os pairs e retorna o embbeding do respectivo CLS'''
    # Iterate over all documents, in batches of size <batch_size>
    for batch_idx in range(0, len(query_pairs), batch_size):

        # Get the current batch of samples
        batch_data = query_pairs[batch_idx:batch_idx + batch_size]

        inputs = tokenizer.batch_encode_plus(batch_data, 
                                       return_tensors='pt',  # pytorch tensors
                                       add_special_tokens=True,  # Add CLS and SEP tokens
                                       max_length = 512, # Max sequence length
                                       truncation = True, # Truncate if sequences exceed the Max Sequence length
                                       padding = True) # Add padding to forward sequences with different lengths
        
        # Forward the batch of (query, doc) sequences
        with torch.no_grad():
            inputs.to(device)
            outputs = model(**inputs)

        # Get the CLS embeddings for each pair query, document
        batch_cls = outputs['hidden_states'][-1][:,0,:]
        
        # L2-Normalize CLS embeddings. Embeddings norm will be 1.
        batch_cls = torch.nn.functional.normalize(batch_cls, p=2, dim=1)
        
        # Store the extracted CLS embeddings from the batch on the memory-mapped ndarray
        embeddings[batch_idx:batch_idx + batch_size] = batch_cls.cpu()
        
    return embeddings



def heat_map(x_labels, y_labels, values):    
    ax = sns.heatmap(values, cmap="Blues")
    ax.set_xticks(np.arange(len(x_labels))+0.4)
    ax.set_yticks(np.arange(len(y_labels))+0.4)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=15,
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=15,
         rotation_mode="anchor")
    
def get_attentions_fig(tokens,attentions):
    '''esta funcao recebe os tokens e o output de cada attention
       12*12 e retorna a relação entre tokens para todas as head da pimeira layer'''
    for i in range(12):
        attention_tokens=attentions[i][0][11].numpy()
        np.shape(attention_tokens)
        # Create a dataset
        df = pd.DataFrame(attention_tokens, columns=tokens)
        # Default heatmap
        p1 = sns.heatmap(df,cmap="BuPu",xticklabels=tokens, yticklabels=tokens)
        plt.title('Attention (Query and RDocument)')
        plt.savefig('attention_RelDoc.png', dpi=100)
        plt.show()
        
def get_pairs(qdocs,cases,ids,detailed_description):
    ''' esta função recebe os cases, documents id e deteilled description
    e retorna pos pairs para o treino e os pairs para o teste, sendo que os
    os ultimos consideram todos os documentos'''
    qdocs['query']=qdocs['query'].astype(str)
    print(qdocs.dtypes)
    queries=list(cases.keys())
    train, test = train_test_split(queries, test_size=0.2)
    ids=np.array(ids)
    pairs=[]
    pairs_test=[]
    for q,doc_id,rel in zip(list(qdocs['query']),list(qdocs['doc_id']),list(qdocs['rel'])):
        idx=np.where(ids == doc_id)
        if str(q) in train:
            pairs.append((cases[str(q)],detailed_description[idx[0][0]]))
        else:
            pass 
            
    for q in test:
        for d in detailed_description:
            pairs_test.append((cases[q],d))
    qdoc_pairs = qdocs[~qdocs['query'].isin(test)]
    return train, test,qdoc_pairs,pairs, pairs_test


def fit(X):
    '''esta função recebe um df com os pairs query id, doc id, label e 
    respectivas features (embbedings) e após implementar uma regressão logistica
    por forma a classificar os documentos, retorna a performance no treino e o coeficientes
    que obtiveream melhor performance.'''
    #treinar com cross validation K=5
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
            X_train, Y_train = Xtrain.iloc[:,:], Xtrain.iloc[:,2]
            X_test, Y_test = Xtest.iloc[:,:], Xtest.iloc[:,2]
            clf = LogisticRegression(random_state=0,C=c,class_weight='balanced').fit(X_train.iloc[:,3:], Y_train)
            coefs_train = clf.coef_[0]
            X_test.iloc[:, 3:] * coefs_train
            X_test['score'] = X_test.iloc[:, 3:].sum(axis=1)
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
    betterc = c_l[np.where(performance==np.amin(performance))]
    clf= LogisticRegression(random_state=0,C=betterc[0],class_weight='balanced').fit(X.iloc[:,3:],X.iloc[:,2])
    return performance, clf.coef_[0]


def evaluate(features,coefs,dropnan = False):
    '''esta funcao recebe um dataframetest e os coeficientes e calcula os socres
    para cada par query/doc. por fim ordena os doc consoante o score para cada
    query, o que permite quaqlquer diferentes metricas, que são retornada por 
    esta funcao.'''
    results_pd=features
    if dropnan == True:
        results_pd.dropna(subset=['rel'], how='all', inplace=True)
    else:
        pass
    results_pd.iloc[:, 2:-1] * coefs
    # print(results)
    results_pd['score']=results_pd.iloc[:, 2:-1].sum(axis=1)
    score_by_query={}
    precision_11point_tv, recall_11point_tv, total_relv_ret_t=[],[],[]
    p10_t, recall_t, ap_t, ndcg_t, mrr_t = 0, 0, 0, 0, 0
    score_total=0
    counter=0
    for caseid in results_pd.A.unique():
        res = results_pd.loc[results_pd["A"] == caseid, ['_id', 'score']]
        res_ord = res.sort_values(by=['score'], ascending=False)
        p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm = eval.eval(res_ord, caseid)
        precision_11point, recall_11point, total_relv_ret = eval.evalPR(res_ord, caseid)  
        p10_t += p10vsm
        recall_t += recallvsm
        ap_t += apvsm
        ndcg_t += ndcg5vsm
        mrr_t += mrrvsm
        precision_11point_tv.append(precision_11point)
        score_by_query[caseid] = [p10vsm, recallvsm, apvsm, ndcg5vsm, mrrvsm]
        counter += 1
        if (np.shape(recall_11point) != (0,)):
            recall_11point_tv.append(recall_11point)      
    score_total = [p10_t / counter, recall_t / counter, ap_t / counter, ndcg_t / counter, mrr_t / counter]
    return score_total,score_by_query,precision_11point_tv,recall_11point_tv 

###############################################################################

## get cases, doc ids & brief summaries
file_names=['cases.bin','doc_ids.bin','brief_summary.bin','detailed_description.bin']
cases,ids,brief_summary,detailed_description=get_data(file_names)

Qrels = "qrels-clinical_trials.txt"
eval = trec.TrecEvaluation(cases, Qrels)
#########   IMPLEMENT BERT to rank our CASES/DOCs pairs        ################

#### TASK 1 INPUT FORMATING TOKENIZATION

## SELECT BERT MODEL AND LOAD IT

model_path = 'dmis-lab/biobert-v1.1'
CLS_token = "[CLS]"
SEP_token = "[SEP]"

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path,  output_hidden_states=True, output_attentions=True)
model = AutoModel.from_pretrained(model_path, config=config).to(device)
# PICKLE IT
pickle.dump(tokenizer, open( "bio_tokenizer.bin", "wb" ) )
pickle.dump(config, open( "bio_config.bin", "wb" ) )
pickle.dump(model, open( "bio_model.bin", "wb" ) )

# Load PICKLE
tokenizer=pickle.load(open( "bio_tokenizer.bin", "rb" ) )
config=pickle.load(open( "bio_config.bin", "rb" ) )
model=pickle.load(open( "bio_model.bin", "rb" ) )


# SELECT PAIR QUERY / RELEVANT DOCUMENT & NON RLEVANT

q_docs = pd.read_csv('qrels-clinical_trials.txt', names=['query','non','doc_id','rel'], skiprows=1, sep='\s+')
non_rel = q_docs[q_docs['rel'] == 0]

q_docs.drop(q_docs[q_docs['rel'] == 0].index, inplace = True)
sample=q_docs.sample() # choose random pandas row 
non_rel = non_rel[non_rel['query']==sample.iloc[0,0]]
sample2=non_rel.sample()

sentence_q = cases[str(sample.iloc[0,0])] # random query
sentence_d1 = detailed_description[ids.index(sample.iloc[0,2])]
sentence_d2 = detailed_description[ids.index(sample2.iloc[0,2])]


## TOKENIZER
inputs_rel,tokens_rel,outputs_rel=sentences_encoder(sentence_q,sentence_d1)
inputs_nrel,tokens_nrel,outputs_nrel=sentences_encoder(sentence_q,sentence_d2)
print(len(tokens_rel))
print(len(tokens_nrel))

# EXTRACT 10 TOKEN and EMBEDDINGS 1st and lst layer
token_10_rel, first_emb_rel,last_emb_rel= extractor_tokens_embeddings(tokens_rel, 10, outputs_rel)
print(token_10_rel)
token_10_nrel, first_emb_nrel,last_emb_nrel= extractor_tokens_embeddings(tokens_rel, 10, outputs_rel)
print(token_10_nrel)

# GET 2 FEATURES Of OUR EMBEDDINGS (USING TSNE)
compute_tsne_plot(token_10_rel,first_emb_rel,last_emb_rel)
compute_tsne_plot(token_10_nrel,first_emb_nrel,last_emb_nrel)

# PLOT HEAT MAP SMILARITY BETWEEN 10 RANDOM CHOOSEN WORDS
tokens_heat,first_emb_heat,last_emb_heat = extractor_tokens_query_docs(tokens_rel,outputs_rel)
cos_similarity_heat=cosine_similarity(first_emb_heat, last_emb_heat) #for our chosen tokens 
heat_map(tokens_heat,tokens_heat,cos_similarity_heat)

tokens_nheat,first_emb_nheat,last_emb_nheat = extractor_tokens_query_docs(tokens_nrel,outputs_nrel)
cos_similarity_nheat=cosine_similarity(first_emb_nheat, last_emb_nheat) #for our chosen tokens 
heat_map(tokens_nheat,tokens_nheat,cos_similarity_nheat)

## Head Attention Analysis
attention = outputs_rel['attentions']
attention_nrel= outputs_nrel['attentions']
get_attentions_fig(tokens_rel,attention)
get_attentions_fig(tokens_nrel,attention_nrel)

'''
call_html()
head_view(attention, tokens_rel) '''

   
# create query/documents pairs
qdocs = pd.read_csv('qrels-clinical_trials.txt', names=['query','non','doc_id','rel'], skiprows=1, sep='\s+')
qdocs.loc[qdocs['rel'] == 2, 'rel'] = 1 # binary
qdocs.drop('non', axis=1, inplace=True)




# Create a memory-mapped numpy array. The array is stored on disk, not on RAM
# The shape argument must match (total number query-doc pairs, CLS embedding size)
# Numpy ndarray that will store (in RAM) the CLS embeddings of each (query, doc) pair
train, test,qdoc_pairs,pairs, pairs_test= get_pairs(qdocs,cases,ids,detailed_description)
np.save('train_cases.npy',train)
np.save('test_cases.npy',test)
np.save('train_pairs.npy',pairs)
np.save('test_pairs.npy',pairs_test)
qdoc_pairs.to_pickle('pd_training_pairs')


# training embbedings
embeddings = np.zeros((len(pairs), 768))
embeddings = extract_cls(pairs, embeddings=embeddings, batch_size=16)
np.save('embedings.npy',embeddings)


# test embbedings 
embeddings_t = np.zeros((len(pairs_test), 768))
embeddings_t = extract_cls(pairs_test, embeddings=embeddings_t, batch_size=16)
np.save('embedings_test.npy',embeddings_t)


# desbloquer para treinar novamente
train_emb=np.load("embedings.npy")
### LOGIT REGRESSION IMPLEMENTATION
qdocs_train=pd.read_pickle('pd_training_pairs')
qdocs_train=qdocs_train.reset_index(drop=True)
# prepare data
embbedings_pd=pd.DataFrame(train_emb)
training_set=pd.merge(qdocs_train, embbedings_pd, left_index=True, right_index=True)
training_set.rename(columns = {'query':'A','doc_id':'_id'}, inplace = True)

# call fit function and get the training performance and best coeficients
performance,coefs=fit(training_set)
np.save('coefs_transformer.npy',coefs)
np.save('performance_training_bert.npy',performance)



# process data creatr pairs_test data frame
coefs=np.load('coefs_transformer.npy')
test=np.load('test_cases.npy')
test_emb=np.load("embedings_test.npy")
test_emb.shape

data_pairs_test=[(q,d) for q in test for d in ids]
len(data_pairs_test)

pairs_test_pd = pd.DataFrame(data_pairs_test, columns =['A', '_id'])
embbedings_test_pd=pd.DataFrame(test_emb)
test_set=pd.merge(pairs_test_pd, embbedings_test_pd, left_index=True, right_index=True)
qdocs.rename(columns = {'query':'A','doc_id':'_id'}, inplace = True)
convert_dict = {'A': str}
qdocs = qdocs.astype(convert_dict)
test_set=test_set.astype(convert_dict)
test_set=pd.merge(test_set, qdocs, on=['A', '_id'],how='left')
test_set

# call evaluate function to get avaliation metrics, using the best coefs
score_model,score_by_query,precision_11point_tv,recall_11point_tv=evaluate(test_set,coefs,dropnan = False)

score_model,score_by_query,precision_11point_tv,recall_11point_tv=evaluate(test_set,coefs,dropnan = True)
score_model


### plot precision recall curve all models
# load metrics
precision_t_vsm=np.load('precision_t_vsm1.npy')
precision_t_lmjm=np.load('precision_t_lmjm1.npy')
precision_t_letor=np.load('precision_t_letor.npy')

precision_11_bert=np.array(precision_11point_tv).mean(axis=0)
precision_11_vsm=np.array(precision_t_vsm).mean(axis=0)
precision_11_lmjm=np.array(precision_t_lmjm).mean(axis=0)
precision_11_letor=np.array(precision_t_letor).mean(axis=0)

recall_11=np.array(recall_11point_tv).mean(axis=0)
recall_11
plt.figure()
plt.xlabel("recall")
plt.ylabel("precision")
plt.plot(recall_11,precision_11_bert/len(test),label="bert")
plt.plot(recall_11,precision_11_vsm/len(test),label="vsm")
plt.plot(recall_11,precision_11_lmjm/len(test),label="lmjm")
plt.plot(recall_11,precision_11_letor/len(test),label="letor")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.show()

