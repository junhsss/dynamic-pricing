import pyodbc
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from hdfs3 import HDFileSystem

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

'''
최종 수정 : 2019.12.17 
버전 : 0.05


추가 수정 내용 : 
[20190618] : loss history, max rank 추가
[20190621] : pyodbc 연결 완료 , from_table 함수에서 lower option 추가
[20190705] : pyodbc 계정 변경 ssg191538 -> pdwbigdata
[20190709] : pymysql (from_mysql 추가)
[20190717] : get_outlier 추가
[20190725] : connection close 수정, autocommit 추가, to_mysql 추가
[20190731] : hdfs connection 추가
[20190802] : hdfs connection 수정
[20190821] : pymysql 수정
[20190910] : hdfs 함수 추가/수정 (self)
[20190919] : hdfs 함수 수정 ( daily_data option)
[20190926] : mysql to if_exist flag 추가
[20191128] : create table 함수 추가 / mysql index 부분 table 이름 수정 
[20191217] : index setting 함수 추가 (mysql), to_mysql 함수 수정


made by 장진규 파트너

'''

class call_data():
    def __init__(self):
        # basic info
        # pdw
        self.server='10.203.9.21,17001'
        self.database='master'
        self.username = 'pdwbigdata' 
        self.password = 'b!gdata100'

        # mysql
        self.path = "mysql+mysqldb://moneymall:"+"moneymall"+"@10.203.5.76/"

        # hdfs
        self.host='hdfs://master001p27.prod.moneymall.ssgbi.com'
        self.port=9000
        self.driver='libhdfs3'
        self.pars={'user':'moneymall'}


    def find_hdfs(self,search_path):
        hdfs = HDFileSystem(host=self.host, port=self.port, driver=self.driver,pars=self.pars)
        
        path_ls=hdfs.ls(search_path)

        print('{} files are found'.format(len(path_ls)))
        return hdfs.ls(search_path)


    def from_hdfs(self, hdfs_path, lower=True,daily_data=False):
        '''
        if daily_data = True --> parquet 파일을 한번 더 찾으러 내려감
        (19-09-01 밑 parquet file) 
        '''
        hdfs = HDFileSystem(host=self.host, port=self.port, driver=self.driver,pars=self.pars)

        if daily_data:
            file_path = hdfs.ls(hdfs_path)[1]
            with hdfs.open(file_path) as f:
                df = pd.read_parquet(f)
        else:
            with hdfs.open(hdfs_path) as f:
                df = pd.read_parquet(f)

        if lower:
            df.columns=list(map(lambda x: x.lower(),list(df.columns)))
        return df

    def from_hdfs_merge(self, hdfs_path, lower=True):
        '''merge all files(concat) in one file path'''
        hdfs = HDFileSystem(host=self.host, port=self.port, driver=self.driver,pars=self.pars)

        df=pd.DataFrame()
        for p in hdfs_path:
            with hdfs.open(p) as f:
                t = pd.read_parquet(f)
            
            df=pd.concat([df,t])

        if lower:
            df.columns=list(map(lambda x: x.lower(),list(df.columns)))
        return df

    def to_hdfs(self, df,save_path,file_format='parquet'):
        '''
        file format can be ... : 'parquet', 'csv'

        '''
        hdfs = HDFileSystem(host=self.host, port=self.port, driver=self.driver,pars=self.pars)

        # if not parquet or pickle -> csv
        with hdfs.open(save_path,mode='wb') as f:
            if file_format=='parquet':
                df.to_parquet(f)
                print('successfully saved as parquet format!')
            else:
                df.to_csv(f)
                print('successfully saved as csv format!')


    def from_table(self,query,lower=True):
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password,autocommit=True)
        try:
            # connect
            df=pd.read_sql(query,cnxn)
            
            if lower:
                df.columns=list(map(lambda x: x.lower(),list(df.columns)))
            return df
        except:
            print('error occured !')
        finally:
            cnxn.close()
            print('cnxn closed !')

    def to_table(self,table_name,df):

        '''
        table_name : ex) scom_userset..jk_test
        df : dataframe  
        '''   
        # iteration setting
        iter_at_once=4200
        k=list(np.arange(0,df.shape[0],iter_at_once))

        if k[-1]<df.shape[0]:
            k.append(df.shape[0])  
        
        # dataframe setting
        # ex) 
        # insert into select col, col, col union all
        #             select col, col, col 
        df.iloc[:,:-1]=df.iloc[:,:-1].applymap(lambda x: str(x)+',')
        df.iloc[:,0]=df.iloc[:,0].apply(lambda x: 'select '+str(x))
        df.iloc[:,-1]=df.iloc[:,-1].apply(lambda x: str(x)+' union all')


        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password,autocommit=True)

        for idx,i in enumerate(k):
            if idx<len(k)-1:        
                my_str=re.sub('\n','',df.iloc[k[idx]:k[idx+1],:].to_string(header=False,index=False))[:-9]
                insert_mystr="insert into {} ".format(table_name)+my_str

                cursor=cnxn.cursor()
                # execute
                cursor.execute(insert_mystr)
        
        cursor.commit()
        cursor.close()
        cnxn.close()
        print('cnxn closed !')

    def create_tmp_table(self,query):
        '''
        pdw에 table 생성
        create를 포함하는 쿼리 날려야함.
        
        ex)
        IF OBJECT_ID('SCOM_USERSET..jk_test') IS NOT NULL
        DROP TABLE jk_test

        CREATE TABLE jk_test
        WITH (CLUSTERED INDEX(ITEM_ID), DISTRIBUTION = HASH(ITEM_ID))
        AS 
        
        select * from ~~

        '''
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+'SCOM_USERSET'+';UID='+self.username+';PWD='+ self.password,autocommit=True)
        try:
            # connect
            cursor = cnxn.cursor()
            cursor.execute(query)
        except:
            print('error occured !')
        finally:
            cnxn.close()
            print('cnxn closed !')



    def from_mysql(self,query,lower=True):
        # at first 
        import pymysql
        pymysql.install_as_MySQLdb()

        
        engine = create_engine(self.path, encoding='utf-8')
        conn = engine.connect()
        
        df = pd.read_sql_query(query, conn)
        if lower:
            df.columns=list(map(lambda x: x.lower(),list(df.columns)))
        
        conn.close()

        print('conn closed!')
        return df



    def to_mysql(self,df,db_name,table_name,index_nm='my_idx',index=False,index_label=None,if_exists_flag='append',upper=False):
        '''
        
        if setting multiple column to index, use col seperate : `
        
        if_exsits flag= ['fail', 'replace', 'append']

        '''
        import pymysql
        from sqlalchemy.types import VARCHAR

        pymysql.install_as_MySQLdb()

        path=self.path+db_name

        engine = create_engine(path, encoding='utf-8')
        conn = engine.connect()

        if upper:
            df.columns=list(map(lambda x: x.upper(),list(df.columns)))
        
        df.to_sql(table_name,conn,if_exists=if_exists_flag,index=index,index_label=index_label)
       
        print('conn closed!')
        conn.close()
    
    def alter_mysql_dtype(self,db_name,table_name,col_ls=[],col_dict={},custom=False):
        '''
        if custom -> col_dict에 입력 ex) {'item_id':'VARCHAR(13)',~~}
        if custom x -> col_ls에 입력 item_id를 제외한 모든 컬럼 float
                       (*item_id : varchar(13))
        '''
        import pymysql
        from sqlalchemy.types import VARCHAR
        from sqlalchemy.types import Float

        pymysql.install_as_MySQLdb()
        path=self.path+db_name
        engine = create_engine(path, encoding='utf-8')
        conn = engine.connect()
        
        query=[]
        if custom:
            for i in col_dict:
                query.append("ALTER TABLE {} modify {} {};".format(table_name,i,col_dict[i]))
        
        else:
            for i in col_ls:
                if i =='item_id':
                    query.append("ALTER TABLE {} modify item_id VARCHAR(13);".format(table_name))
                else:
                    query.append("ALTER TABLE {} modify {} Float;".format(table_name,i))
                
        
        for i in query:
            conn.execute(i)        

        print('conn closed!')
        conn.close()

    def set_mysql_index(self,db_name,table_name,index_nm='my_idx',index_col=False):
        '''
        if setting multiple column to index, use col seperate : `
        '''
        import pymysql
        from sqlalchemy.types import VARCHAR

        pymysql.install_as_MySQLdb()

        path=self.path+db_name

        engine = create_engine(path, encoding='utf-8')
        conn = engine.connect()
        conn.execute('CREATE INDEX  `{}` ON {} ({});'.format(index_nm,table_name,index_col))
        
        print('conn closed!')
        conn.close()

class feature_engineering():
    
    def __init__(self,df):
        self.df=df
        # 기준 잡기
        #threshold=0.7

    def delete_high_cor_var(self,threshold):
        from collections import Counter
        deleted=[]
        over_k=True
        
        while over_k:

            # corr matrix
            corr_matrix = self.df.corr().abs()
            high_corr_var=np.where(corr_matrix>threshold)
            high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

            # threshold 보다 높은 col이 없으면 stop
            if len(high_corr_var)==0:
                break
            else:
                # 가장 많이 겹치는 컬럼 제거 
                high_corr_var_dict=Counter([j for i in high_corr_var for j in i])
                delete_candi=max(high_corr_var_dict, key=high_corr_var_dict.get)
                self.df=self.df.drop(delete_candi,axis=1)
                deleted.append(delete_candi)
        
        print('제거된 컬럼 :',deleted)
        return self.df

    def delete_high_cor_var_any(self,threshold):
        # 둘다 제거
        corr = self.df.corr()
        m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > threshold).any()
        self.df=self.df.loc[:,m]
        return self.df

    def corr_plot(self,w,h):
        corr=self.df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        f=plt.figure(figsize=(w,h))
        sns.heatmap(corr,cmap="coolwarm",mask=mask)
        return f

    def get_sim_col_re(self,keyword,forward=True):
        # pandas에서 특정 키워드를 공유하는 변수들을 selection함
        # forward=False일경우 뒤에서 부터
        lst_len=len(keyword)
        if forward:
            candi_list=list(map(lambda x: x[:lst_len]==keyword, list(self.df.columns)))
        else:
            candi_list=list(map(lambda x: x[-lst_len:]==keyword, list(self.df.columns)))
        col=[i for i,j in zip(self.df.columns,candi_list) if j==True]
        return col

    def max_rank(x):
        ## top 1
        from collections import Counter
        return Counter(x).most_common(1)[0][0]


    def get_outlier(descr):
        # boxplot base outlier finder
        ## input : describe
        qt1,qt2=descr[4],descr[6]
        iqr= (qt2-qt1)*1.5
        out_p=qt2+iqr
        out_m=qt1-iqr
        return out_p,out_m


class Modeling():
    ### scikit-learn base
    
    def __init__(self, model,X_train,y_train,X_test,y_test):
        self.model=model
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        
        self.train_shape=X_train.shape
        self.test_shape=X_test.shape
        
    
    def fitting(self):
        self.model.fit(self.X_train,self.y_train)
    
    def cross_validation(self,cv,scoring=None):
        cv_results=cross_val_score(self.model,self.X_train,self.y_train, n_jobs=-1,cv=cv,scoring=scoring)
        return cv_results
            
    def prediction(self,test):
        ## new data
        pred_test=self.model.predict(test)
        return pred_test
        #print(classification_report(self.y_test,pred_y_test))
    
    def get_feature_importance(self,w,h):
        ### w: width, h: height
        features = self.X_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        f=plt.figure(figsize=(w,h))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='gray', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        return f

    def loss_history(history):
        ### for keras

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