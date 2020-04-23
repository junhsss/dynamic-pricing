import sys
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

from helper_module.data_science import call_data

path = '/data01/program/anaconda3/notebook/jk/'
sys.path.append(path)

class DanawaParser():
    def __init__(self):
        self.numdays=1
        self.min_time=0
        self.max_time=9
        self.base = datetime.now()
        self.ls = self.get_file_ls_run(self.numdays,self.min_time,self.max_time)
        self.ca = call_data()

    def date_str(self,date_time):
        to_str=lambda x: '0'+str(x) if x<10 else str(x)
        month=to_str(date_time.month)
        day=to_str(date_time.day)
        hour=to_str(date_time.hour)        
        date_str=str(date_time.year)+month+day
        return date_str

    def file_nm(self,now_str,min_time,max_time):        
        file_ls=[]
        for i in range(min_time,max_time):
            if i<10:
                i='0'+str(i)
            else:
                i=str(i)

            for j in ['0000','3000']:

                r='EP_'+now_str+i+j+'.parquet'
                file_ls.append(r)
        return file_ls

    def get_file_ls_run(self,numdays,min_time,max_time):
        dnw_ls = []
        for x in range(numdays):
            now_str = self.date_str(self.base - pd.Timedelta(days=x))
            dnw_ls.append(self.file_nm(now_str,min_time,max_time))

        dnw_ls = np.array(dnw_ls).reshape(-1)
        return dnw_ls

    def read_danawa(self,x,meta=True):
        if meta:
            file = self.ca.from_hdfs('/moneymall/CouponFeature/Danawa/meta/'+x)
        else:
            file = self.ca.from_hdfs('/moneymall/CouponFeature/Danawa/prc_detail/'+x)
        file['dt']=x[3:7]+'-'+x[7:9]+'-'+x[9:11]
        file['hour']=x[11:15]
        file=file.rename(columns={'ssgitemids':'item_id'})
        return file