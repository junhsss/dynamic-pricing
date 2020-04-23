import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper_module.data_science import call_data
import warnings
import datetime
from collections import Counter

from statsmodels.api import GLM
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor


def mode(x):
    # 최빈값이 하나보다 많다면 list를 반환
    counts = Counter(x)
    max_count = max(counts.values())
    results = [x_i for x_i, count in counts.items() if count == max_count]
    
    return results[0]

def preprocessing_df(prom_offer,ord_item,merge_flag=True):
    if merge_flag:
        df=prom_offer.merge(ord_item,on=['orord_no','orord_item_seq'],how='right')
 
    else:
        df=ord_item.copy()
    df['date_info']=pd.to_datetime(df.ord_dt.astype('str')+'-'+df.ord_hm)
    df.date_info=df.date_info-datetime.timedelta(hours=10)
    df.date_info=df.date_info.astype('str').apply(lambda x: x[:10])
    
    # - qty -> abs
    df['ownco_bdn_firs_dc_amt']=df.ownco_bdn_firs_dc_amt/df.ord_qty.abs()
    
    # 실주문
    rorord_no=df.groupby(['orord_no']).ord_amt.sum().reset_index()
    rorord_dict=dict(rorord_no.values)

    dff=df[df.orord_no.map(rorord_dict)>0]
    
    if merge_flag:
        dff.loc[:,['prom_id','offer_id']]=dff.loc[:,['prom_id','offer_id']].fillna('9999999999')
        
    return dff

def agg_df(df_old):
    '''
    preprocessing_df results input
    '''
    dff_old=df_old.groupby(['item_id','date_info']).agg({'ownco_bdn_firs_dc_amt':mode,'coopco_bdn_firs_dc_amt':mode,'ord_qty':sum,'ord_amt':sum,'sellprc':mode,'splprc':mode})
    dff_old['dc_rate']=dff_old.eval('(ownco_bdn_firs_dc_amt/(ord_amt/ord_qty))*100').round(2)
    dff_old=dff_old.reset_index()
    dff_old['ds']=dff_old.date_info.apply(lambda x: x[-5:])
    
    dff_old['test_yn']=dff_old.ds.apply(lambda x: 1 if int(x[:2]+x[-2:])>=1002 else 0)
    #dff_old['profit']=dff_old.eval('ord_amt-(splprc+ownco_bdn_firs_dc_amt+coopco_bdn_firs_dc_amt)*ord_qty')
    dff_old['profit']=dff_old.eval('(sellprc-(splprc+ownco_bdn_firs_dc_amt+coopco_bdn_firs_dc_amt))*ord_qty')

    dff_old['profit_ratio']=dff_old.eval('profit/(splprc*ord_qty)')
    dff_old['margin_ratio']=dff_old.eval('(sellprc-ownco_bdn_firs_dc_amt-coopco_bdn_firs_dc_amt-splprc)/splprc')
    
    return dff_old

def results_eval(x):
    if x[-3]>0:          # 이익
        if x[-2]>0:      #이익률 
            if x[-1]>0:  #매출
                r=1     # CASE 1 이익 + 이익률 + 매출 +
            else:
                r=2     # CASE 2 이익 + 이익률 + 매출 -
        else:            
            r=3         # CASE 3 이익 + 이익률 - 
    else:
        if x[-2]<0:
            if x[-1]>0:                
                r=4           # 이익 - 이익률 - 매출 +
            else:
                r=6           # 이익 - 이익률 - 매출 -
        else:
            r=5               # CASE 5 이익 - 이익률 + 매출 -

    return r



def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_nom=np.array(list(map(lambda x: 1 if x==0 else x,y_true)))
    
    results=np.mean(np.abs((y_true - y_pred) / (y_true_nom))) * 100
    return results


def median_ape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_nom=np.array(list(map(lambda x: 1 if x==0 else x,y_true)))
    
    results=np.median(np.abs((y_true - y_pred) / (y_true_nom))) * 100
    return results


def mape_func(x):
    return mape(x['pred_ord_cnt'],x['ord_cnt'])

def median_ape_func(x):
    return median_ape(x['pred_ord_cnt'],x['ord_cnt'])
    


def itemby_eval(pred,y_train,training_seq):
    # if train set -> training seq
    # if test set -> n_test
    y_train_itemby=y_train[:,-1].reshape(-1,training_seq)

    pred_itemby=pred[:,-1].reshape(-1,training_seq)

    itemby_pred_results=(np.mean(np.square(pred_itemby - y_train_itemby),1)) 
    return itemby_pred_results



#####################
class PriceElasticity():
    '''
    prince elasticity class

    using GLM, RANSAC LM

    '''
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
        ransac=RANSACRegressor()
        ransac.fit(item_df.loc[:,self.col].values.reshape(-1,1),np.log(item_df.ord_qty))

        pred=ransac.predict(item_df.loc[:,self.col].values.reshape(-1,1))

        params=[ransac.estimator_.intercept_,ransac.estimator_.coef_[0]]
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
            param,pred2=self.ransac_fit(item_df)            
            sub=sns.scatterplot(item_df.loc[:,self.col],np.log(item_df.ord_qty),ax=ax[idx])
            sns.lineplot(item_df.loc[:,self.col],pred,ax=ax[idx])
            sns.lineplot(item_df.loc[:,self.col],pred2,ax=ax[idx])
            
        return f,ax


######
class PRICE_OPT_RESULT():
    def __init__(self,real):
        self.except_ls=[]       
        self.df=self.add_variable(real)
        self.item_ls=real.item_id.unique()
        self.std_ctg_id_ls=real.std_ctg_id.unique()
        self.std_ctg_mcls_id_ls=real.std_ctg_mcls_id.unique()
        
        # date setting
        self.comp_dt='2019-12-05'
        self.test_dt='2019-12-20'
        self.test_range = 7
        
        
        self.comp_dt_end=str(pd.to_datetime(self.comp_dt)+pd.Timedelta('{}D'.format(self.test_range)))[:10]
        self.test_dt_end=str(pd.to_datetime(self.test_dt)+pd.Timedelta('{}D'.format(self.test_range)))[:10]
        
             
        self.agg={ 'ord_qty':sum,  'ord_amt':sum,'dc_amt':sum,'item_prft_amt':sum,'sellprc':max, # 'splprc':max,
                   'item_firs_dc_amt':sum,'item_scnd_dc_amt':sum,'offer_300_dc_amt':sum,
                   'profit_ratio':np.mean,'item_frst_dc_ratio':np.mean,'item_scnd_dc_ratio':np.mean} 
    
        print(len(self.item_ls), 'item 개수')
        print(len(self.std_ctg_id_ls), 'std_ctg_id 개수')
        print(len(self.std_ctg_mcls_id_ls), 'std_ctg_mcls_id 개수')
        
    def print_test_info(self):
        print('비교 기간 시작 / 종료: ',self.comp_dt,' to ', self.comp_dt_end)
        print('테스트 기간 시작 날짜 : ',self.test_dt,' to ', self.test_dt_end)
        print('테스트 기간 : ',self.test_range)

    def add_variable(self,dff):
        dff['item_prft_amt']=dff.eval('(ord_amt-splprc*ord_qty-dc_amt)*(1/vatrt)')
        dff['profit_ratio']=dff.eval('item_prft_amt/((ord_amt-dc_amt)*(1/vatrt))')
        return dff
        
    def test_date_setting(self,dff):
        dff.dt=dff.dt.astype('str')
        dff['test_yn']=np.where(dff.dt>=self.test_dt,1,0) 
        
        
        dff = dff.query('(dt<="{0}"&dt>="{1}")|(dt<="{2}"&dt>="{3}")'.format(self.comp_dt_end,self.comp_dt,
                                                                                 self.test_dt_end,self.test_dt))
        return dff
    
    def add_diff_agg(self,dff_agg):
        dff_agg=dff_agg.fillna(0)
        dff_agg['diff_ord_amt']=dff_agg.loc[:,'ord_amt'][1]-dff_agg.loc[:,'ord_amt'][0]
        dff_agg['diff_item_prft_amt']=dff_agg.loc[:,'item_prft_amt'][1]-dff_agg.loc[:,'item_prft_amt'][0]
        return dff_agg
    
    
    def get_same_ctg_df(self):
        query ='''
        select ord_dt as dt, *
        from scom_userset..coupon_ord_test
        where 
            std_ctg_id in {0}
        and item_id not in {1}
        '''.format(tuple(self.std_ctg_id_ls),tuple(self.item_ls))
        
        ctg_df = ca.from_table(query)
        ctg_df=self.add_variable(ctg_df)       
        return ctg_df
    
    #####################
    ##### outputs
    #####################
    # r2_preprocessing
    def summary_table(self,dff):
        dff=self.test_date_setting(dff)
        dff=dff.groupby(['test_yn']).agg(self.agg).reset_index()

        dff_t=dff.round(3).astype('str').transpose()
        dff_t['ratio']=((dff.iloc[1,:]/dff.iloc[0,:])-1)*100
        dff_t.ratio=dff_t.ratio.round(2)
        return dff,dff_t
        
    def summary_table_itemwise(self,dff):
        dff=self.test_date_setting(dff)
            
        df_filter=dff[dff.item_id.isin(self.except_ls)==False]
        itemwise=df_filter.groupby(['test_yn','item_id']).agg(self.agg)

        itemwise1=itemwise.reset_index()
        itemwise2=itemwise.unstack(0)
        itemwise2=self.add_diff_agg(itemwise2)
            
        col=['test_yn','item_id']+list(self.agg.keys())
        itemwise1=pd.DataFrame(itemwise1.values,columns=col)

        
        
        
        return itemwise1,itemwise2