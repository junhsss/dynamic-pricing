import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from datetime import datetime
from statsmodels.api import GLM
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor

from helper_module.data_science import feature_engineering
from helper_module.data_science import call_data
import sys
path = '/data01/program/anaconda3/notebook/jk/'
sys.path.append(path)

#############################
#### read pg file
#############################

def setting_grouping_file():
    now = datetime.now()
    
    month_str = lambda x: '0'+str(x) if x<10 else str(x)
        
    if now.day<=15:
        now=datetime(now.year,now.month,1)- dt.timedelta(days=1)
        export_file_nm=month_str(now.month)+'_2'
    else:
        now=datetime(now.year,now.month,15)
        export_file_nm=month_str(now.month)+'_1'

    past = now-pd.Timedelta('120D')
    date_1 = get_date_str(now)
    date_2 = get_date_str(past)
    
    return export_file_nm,date_1,date_2

##############################
##### cleansing
##############################
def get_date_str(date_time):
    yyyy=str(date_time.year)
    mm=str(date_time.month)
    dd=str(date_time.day)

    if len(mm)==1:
        mm='0'+mm
    if len(dd)==1:
        dd='0'+dd

    date_str=yyyy+'-'+mm+'-'+dd
    return date_str

def merge_data(lss,g=0,only_group=True):
    # path에서 마지막 파일을 읽어서 합쳐줌 (1일)
    merged=pd.DataFrame()
    for ls in lss:
        for i in ls:
            daily=ca.from_hdfs(i,daily_data=True)

            if only_group:
                daily=daily.merge(g)
            else:
                pass
            merged=pd.concat([merged,daily])
    return merged


def fill_empty_rows(dff,generate_today=False):
    '''
    모든 값이 0이어서 row가 없는 아이템들을 일자별로 채워줌
    generate_today=True 일경우 가장 최근 날짜 값도 추가로 생성 (X 데이터가 없는)
    live test시 True, 아닐경우 False로 놓고 돌리기
    '''
    # generate_today
    now = datetime.now()
    today = get_date_str(now)
    

    item_ls=dff.item_id.unique()
    date_ls=dff.dt.unique()

    if generate_today:
        date_ls = list(date_ls)+[today]
    else:
        pass


    item_mul = np.repeat(item_ls,len(date_ls))
    date_mul = list(date_ls)*len(item_ls)

    empty=pd.DataFrame({'item_id':item_mul,'dt':date_mul})

    expected =  len(date_ls)*len(item_ls)
    results = dff.merge(empty,how='right')
    
    print('unique date x item = expected rows \n ==>',str(len(date_ls))+'x'+str(len(item_ls))+' =',str(expected))
    print('results : ',results.shape[0])
    
    return results

def diff_zero_sales(df):
    df.dt=pd.to_datetime(df.dt)
    df['dt_shift']=df.groupby('item_id').apply(lambda x: x.loc[:,['dt']]-x.loc[:,['dt']].shift(1))
    df.dt_shift=df.dt_shift.fillna(0).apply(lambda x: str(x)[0])
    return df

def fill_zero_prc(df):
    '''
    sell_prc, best_prc -> ffill, backfill 로 na 값 채워줌
    '''
    filled=df.loc[:,['item_id','sell_prc','best_prc']].groupby('item_id').fillna(method='ffill')

    df.loc[:,['sell_prc','best_prc']]=filled

    b_filled=df.loc[:,['item_id','sell_prc','best_prc']].groupby('item_id').fillna(method='backfill')

    df.loc[:,['sell_prc','best_prc']]=b_filled
    
    return df

##############################
##### split data
##############################
def save_train_test_data_simple(last_fname,outputs):
    
    path='/data01/program/anaconda3/notebook/jk/price_opt_v2/rawdata'

    pd.to_pickle(outputs,'{}/outputs_{}.sav'.format(path,last_fname))
    print('save completed !')

def read_train_test_data_simple(last_fname):
    path='/data01/program/anaconda3/notebook/jk/price_opt_v2/rawdata'    

    outputs=pd.read_pickle('{}/outputs_{}.sav'.format(path,last_fname))
    print('load completed !')
    
    return outputs


#######################
######## split data
#######################
def num_of_nine(x):
    num_of_9=[1 if int(i)==9 else 0 for i in str(int(x))]            
    return sum(num_of_9)

def price_as_str(x,idx=0):
    try:
        result=str(int(x))[idx]
    except:
        result=-1
    return int(result)

def get_dc_derived_var(dc_ods,sellprc):
    # 순서 
    # 'dc_ods', 'dc_3', 'dc_add', 'bestprc', 'sellprc', 'n_plus', 'num_of_9', 'price_0', 'price_1', 'price_2'
    dc_add=dc_ods
    dc_3=0
    bestprc=(np.exp(sellprc)-1).round()*(1-dc_ods*0.01)
    n_plus=0
    num_of_9=num_of_nine(bestprc)
    price_0=price_as_str(bestprc,idx=0)
    price_1=price_as_str(bestprc,idx=1)
    price_2=price_as_str(bestprc,idx=1)
    
    return dc_ods,dc_3,dc_add,np.log(bestprc).round(3),np.log(sellprc).round(3),n_plus,num_of_9,price_0,price_1,price_2


class SplitData():
    def __init__(self):        
        self.start=0
        self.window=7
        self.n_test = 7
        self.n_features = None
        self.uniq_item_count = None
        self.n_var=1
        
    def get_train_test(self,one):
        shape=one.values.shape[0]-self.window
        #n_var+=1
        x_ls,y_ls=[],[]

        s=self.start
        for i in range(shape):
            target_stamp=np.array(list(one.values[s+self.window,1:(self.n_var+1)])+[0]*(self.n_features-self.n_var))
            # old (discount : 2rd col) np.array([0,one.values[s+window,1]]+[0]*(n_features-2)) 

            # idx 맞추기 위해 s+1, window +1
            x_reshaped=list(np.concatenate([one.values[s+1:s+self.window,1:].reshape(-1),target_stamp.reshape(-1)]))

            # idx 맞추기 위해 s+1, window +1
            y_reshaped=list(one.values[s+1:s+self.window+1,0].reshape(-1))

            x_ls.append(x_reshaped)
            y_ls.append(y_reshaped) 
            s+=1

        X=np.array(x_ls)#.reshape(-1,window,n_features)
        Y=np.array(y_ls)#.reshape(-1,window+1)
        return [X,Y]

    def get_train_test_run(self,one):
        get_xtr=lambda x: x[0][:-self.n_test+1]
        get_xte=lambda x: x[0][-self.n_test:]

        get_ytr=lambda x: x[1][:-self.n_test+1]
        get_yte=lambda x: x[1][-self.n_test:]


        # min count
        min_count=one.groupby('item_id').y.count().min()

        grouped=one.groupby('item_id').tail(min_count).groupby('item_id').apply(self.get_train_test).reset_index()

        X_train=np.array(list(grouped[0].apply(get_xtr).values)).reshape(-1,self.window,self.n_features)
        X_test=np.array(list(grouped[0].apply(get_xte).values)).reshape(-1,self.window,self.n_features)

        y_train=np.array(list(grouped[0].apply(get_ytr).values)).reshape(-1,self.window)
        y_test=np.array(list(grouped[0].apply(get_yte).values)).reshape(-1,self.window)

        training_seq_len=X_train.shape[0]/self.uniq_item_count
        test_seq_len=X_test.shape[0]/self.uniq_item_count

        head=min_count-test_seq_len

        item_idx_tr=one.reset_index().groupby('item_id').head(head).groupby('item_id').tail(training_seq_len).item_id
        item_idx_te=one.reset_index().groupby('item_id').tail(test_seq_len).item_id

        print('seq length : ',min_count)
        print('training seq length : ',training_seq_len)
        print('test seq length : ',test_seq_len)
        print('\n')
        print(X_train.shape,X_test.shape)
        print(y_train.shape,y_test.shape)

        return X_train,X_test,y_train,y_test,item_idx_tr,item_idx_te


#############################
######## simul
#############################
class GetSimul():
    def __init__(self):
        self.window=None
        self.n_features=None
        self.n_test=None
        self.item_idx_te=None
        self.g=None

        
    def for_eval_part1(self,y_test,pred_test,for_eval):
        # return dt,item_id, model, pred
        pred_results=pd.DataFrame({'item_id':self.item_idx_te,'pred':pred_test[:,-1],'target':y_test[:,-1]})
        pred_results=pred_results.reset_index(drop=True)

        for_eval=for_eval.groupby(['item_id']).tail(self.n_test).reset_index(drop=True)

        dm_pred=pd.concat([for_eval.loc[:,['dt']],pred_results.loc[:,['item_id','pred']]],axis=1)
        dm_pred['model']='lstm'
        dm_pred['g']=self.g
        #dm_pred=dm_pred.rename(columns={'pred':'pred_ord_cnt','model':'MODEL'})
        dm_pred=dm_pred.iloc[:,[0,1,3,4,2]]
        return dm_pred

    def cal_simul_old(self,simul_item):
        r=np.array([]).reshape(-1,self.window,self.n_features)

        for i in range(35):
            simul_sample=simul_item
            simul_sample=simul_sample.reshape(-1,self.window,self.n_features)
            simul_sample[0][-1][1]=i
            r=np.concatenate([r,simul_sample])

        return r

    def cal_simul(self,simul_item):
        #### simul 계산
        r=np.array([]).reshape(-1,self.window,self.n_features)
                
        for i in range(35):
            simul_sample=simul_item
            simul_sample=simul_sample.reshape(-1,self.window,self.n_features)
            sellprc=simul_sample[0][-2][4]
            
            # 조정 값
            derived_var_ls=get_dc_derived_var(i,sellprc)
            for idx,j in enumerate(derived_var_ls):
                simul_sample[0][-1][idx]=j
                
            #simul_sample[0][-1][1]=i
            r=np.concatenate([r,simul_sample])
        return r


    def get_simul_df(self,X_test,for_eval,model):
        X_test_simul=np.array(list(map(self.cal_simul,X_test)))
        X_test_simul_reshape=X_test_simul.reshape(-1,self.window,self.n_features)

        simul_pred=model.predict(X_test_simul_reshape)


        print('X test simul shape :',X_test_simul.shape)
        print('simul pred shape :',simul_pred.shape)

        for_eval=for_eval.groupby(['item_id']).tail(self.n_test).reset_index(drop=True)

        uniq_date=for_eval.dt.sort_values().unique()[-self.n_test:]


        # make df
        simul_df=pd.DataFrame({'dt':np.tile(np.repeat(uniq_date,35),int(X_test_simul.shape[0]/self.n_test)),
                           'item_id':np.repeat(self.item_idx_te.values,35),
                           'model':'lstm',
                           'g':self.g,
                           'dc_ods':np.tile(np.arange(35),X_test_simul.shape[0]),
                           'pred_ord_cnt':simul_pred[:,-1], 
                          })


        # post processing 
        #simul_df.pred_ord_cnt=np.where(simul_df.pred_ord_cnt<0,0,simul_df.pred_ord_cnt)
        simul_df.pred_ord_cnt=simul_df.pred_ord_cnt.apply(lambda x: 0 if x<0 else x)

        return simul_df

##############################
##### model eval
##############################
def postprocessing(pred):
    # negative pred -> 0
    pred_processed = np.where(pred<0,0,pred)
    
    return pred_processed


def loss_history(history):
    f=plt.figure(figsize=(10,5))
    y_val_loss=history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_val_loss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    return f


def qty_pred_pois(x):
    flag = True
    x=int(x)
    
    filter_r=np.array([])
    while flag:
        r=np.random.poisson(lam=0.5883,size=x)
        filter_r=np.append(filter_r,np.extract(r>0,r))
        
        if len(filter_r)>=x:
            flag=False
            break
    return np.sum(filter_r[:x])

def merge_pois_df():
    path='/data01/program/anaconda3/notebook/jk/price_opt/rawdata'

    for_eval=pd.read_pickle('{}/for_eval.sav'.format(path))
    for_eval=for_eval.groupby(['item_id']).tail(22).reset_index(drop=True)

    pred_results=pd.read_pickle('rawdata/pred_results.sav').reset_index(drop=True)

    results=pd.concat([pred_results.loc[:,['item_id','pred']],for_eval.loc[:,['date_info','qty','y']]],axis=1)
    
    return results


def for_eval_part1_analytics(item_idx_te,y_test,pred_test,for_eval,n_test):
    # return dt,item_id, model, pred
    pred_results=pd.DataFrame({'item_id':item_idx_te,'pred':pred_test[:,-1],'target':y_test[:,-1]})
    pred_results=pred_results.reset_index(drop=True)
    
    
    #for_eval=pd.read_pickle('rawdata/for_eval.sav')
    for_eval=for_eval.groupby(['item_id']).tail(n_test).reset_index(drop=True)
    
    dm_pred=pd.concat([for_eval.loc[:,['date_info','qty','y']],pred_results.loc[:,['item_id','pred']]],axis=1)
    dm_pred.to_pickle('rawdata/dm_pred_a.sav')
    
    return dm_pred


#######################
#### price elasticity 
#######################

class PriceElasticity():
    def __init__(self):
        self.col='dc_ratio' 

    def glm_fit(self,item_df):
        glm=GLM(endog=np.log(item_df.ord_qty),exog=sm.add_constant(item_df.loc[:,self.col]))
        glm_results=glm.fit()
        if glm_results.converged!=True:
            print('not converged')

        pred=glm_results.predict(sm.add_constant((item_df.loc[:,self.col])))
        return glm_results.params.values,pred

    def ransac_fit(self,item_df):
        ransac=RANSACRegressor(max_trials=300,random_state=1234)
        try:
            ransac.fit(item_df.loc[:,self.col].values.reshape(-1,1),np.log(item_df.ord_qty))
            pred=ransac.predict(item_df.loc[:,self.col].values.reshape(-1,1))
            params=[ransac.estimator_.intercept_,ransac.estimator_.coef_[0]]
        except:
            print('ransac not converged !')
            params=np.array([0,0])
            pred=np.array([0]*len(item_df))
            
        return params,pred
    
        
    def elasticity(self,x0,x1,p1,p2,print_out=True):    
        q1=x1*p1+x0
        q2=x1*p2+x0

        delta_p=(p2-p1)/p1
        delta_q=(q2-q1)/q1
        ela=round(delta_q/delta_p,2)

        if print_out:
            print(' delta p -> delta q \n  ', round(delta_p,2),'  -> ', round(delta_q,2))
            print(' elasticity :', ela)
        return np.abs(ela)
    
    def get_elasticity_median(self,item_df,param,print_out=True):
        #param,_=self.glm_fit(item_df)
        x0,x1=param[0],param[1]
        
        e=[]
        for i in list(range(65,101)):
            e.append(self.elasticity(x0,x1,i,i-1,print_out=False))
        if print_out:
            print('elasticity median ({}): '.format(self.col),np.median(e))
        return e

    def glm_plotting(self,item_df,pred):
        #_,pred=self.glm_fit(item_df)
        
        f=plt.figure(figsize=(15,5))
        sub=sns.scatterplot(item_df.loc[:,self.col],np.log(item_df.ord_qty))
        sns.lineplot(item_df.loc[:,self.col],pred)
        return f,sub

    def plotting_all(self,item_df):
        #_,pred=self.glm_fit(item_df)
        
        f,ax=plt.subplots(nrows=3,ncols=1,figsize=(10,12))
        for idx,i in enumerate(['frst_dc_ratio','scnd_dc_ratio','dc_ratio']):
            
            self.col=i
            
            param,pred=self.glm_fit(item_df)
            try:
                param,pred2=self.ransac_fit(item_df)            
            except:
                param=np.array([0,0])
                pred2=[0]
            sub=sns.scatterplot(item_df.loc[:,self.col],np.log(item_df.ord_qty),hue=item_df.test_yn,ax=ax[idx])
        
            sns.lineplot(item_df.loc[:,self.col],pred,ax=ax[idx],color='black')   # <- glm
            sns.lineplot(item_df.loc[:,self.col],pred2,ax=ax[idx],color='gray') # <- ransac
    
        return f,ax