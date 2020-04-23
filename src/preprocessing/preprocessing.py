import numpy as np
import random
import pandas as pd
import os
import tqdm
import multiprocessing
from pathlib import Path
import copy
import warnings
warnings.filterwarnings("ignore")

from helper_module.data_science import call_data


class Preprocessor:
    def __init__(self, itemid):
        self.itemid = itemid
        visit, price = self.query(itemid)
        visit, price = self.sort_by_date(visit, price)
        visit, price, self.configuration = self.feature_selecter(visit, price)
        visit, price = self.impute_zeros(visit, price)
        
        merged = self.merge_visit_and_price(visit, price)
        
        merged = self.feature_engineering(merged)
        
        merged = self.reward_engineering(merged)
        
        self.merged = self.log_transform(merged)
        """
        training, validation = self.split(merged)
        training, validation, mean = self.normalizer(training, validation)

        self.training = training
        self.validation = validation
        """

    def query(self, itemid):
        """
         'critn_dt', 기준일자
         'item_id',
         'std_ctg_mcls_id', 중
         'std_ctg_scls_id', 소
         'std_ctg_dcls_id', 세
         'item_reg_div_cd', 상품등록구분코드
         'item_sell_type_cd', 상품판매유형코드
         'sell_stat_cd', 판매상태코드
         'lnkd_spl_ven_id', 업체ID
         'brand_id', 브랜드ID
         'b2c_apl_rng_cd', B2C적용범위코드
         'pc_sellprc', PC 판매가
         'pc_dc_prc', PC 할인가
         'pc_lwst_sellprc', PC 최저판매가
         'mobil_sellprc',
         'mobil_dc_prc',
         'mobil_lwst_sellprc',
         'splprc', 공급가
         'sellprc', 판매가
         'ord_occ_yn', 주문발생여부
         'ord_qty', 주문수량
         'ord_amt', 주문금액
         'dc_amt', 할인금액
         'rlord_amt', 실주문금액
         'prc_opti_rplc_dc_amt', 가격최적화대체할인
         'spl_amt', 공급가
         'prft_amt', 이익금액
         'prc_opti_prft_amt', 가격최적화이익금액
         'ord_mbr_cnt', 주문회원수
         'recom_reg_cnt', 상품평등록수
         'recom_avgscr', 상품평평점
         'load_dts'
        """
        visit = df[df.item_id.isin([itemid])]
        price = price_df[price_df.item_id.isin([itemid])]
        """
        LEGACY
        price = ca.from_table(
            f"SELECT * FROM SCOM_MINE..EADD_ITEM_PRC_OPTI_VAR WHERE ITEM_ID = '"
            + itemid
            + "'ORDER BY 1 DESC"
        )
        """

        return visit, price


    def sort_by_date(self, visit, price):
        visit["dt"] = pd.to_datetime(visit["dt"])
        visit.sort_values("dt", inplace=True, ascending=False)

        price["critn_dt"] = pd.to_datetime(price["critn_dt"])
        price.sort_values("critn_dt", inplace=True, ascending=False)

        return visit, price

    def feature_selecter(self, visit, price):
        visit_features = [
            "dt",
            "item_id",
            "cart_cnt",
            "etc_ckw_uv",
            "price_site_ckw_uv",
            "ma_uv",
            "mw_uv",
            "pc_uv",
            "srchwd_cnt",
            "srch_uv",
        ]

        price_features = [
         'critn_dt',
         'item_id',
         'mobil_sellprc',
         'mobil_dc_prc',
         'mobil_lwst_sellprc',
         'ord_occ_yn',
         'ord_qty',
         'ord_amt',
         'dc_amt',
         'rlord_amt',
         'prft_amt',
         'ord_mbr_cnt',
         'recom_reg_cnt',
         'recom_avgscr',
        ]
        
        ctg_mcls_id = price.std_ctg_mcls_id.value_counts().index[0]
        ctg_scls_id = price.std_ctg_scls_id.value_counts().index[0]
        ctg_dcls_id = price.std_ctg_dcls_id.value_counts().index[0]
        
        
        configuration = pd.DataFrame({
                                      "ctg_mcls_id":[ctg_mcls_id],
                                      "ctg_scls_id":[ctg_scls_id],
                                      "ctg_dcls_id":[ctg_dcls_id],
                                     })
        
        return visit[visit_features], price[price_features], configuration

    def impute_zeros(self, visit, price):
        """
        사용하는 Feature가 달라질 때는 고려하여 사용.
        """
        visit.fillna(0, inplace=True)
        price.fillna(0, inplace=True)

        return visit, price


    def merge_visit_and_price(self, visit, price):
        merged = visit.merge(price, how="outer", left_on="dt", right_on="critn_dt")
        merged = merged[
            merged.dt.isin(pd.date_range(start="2019-10-01", end="2020-03-15"))
        ]
        """
        item_id = merged.item_id_x.value_counts().index[0]
        ctg_mcls_id = merged.std_ctg_mcls_id.value_counts().index[0]
        ctg_scls_id = merged.std_ctg_scls_id.value_counts().index[0]
        ctg_dcls_id = merged.std_ctg_dcls_id.value_counts().index[0]
        """

        merged.drop(["critn_dt","item_id_x", "item_id_y","dt"],\
                     1, inplace=True)

        return merged


    def feature_engineering(self, merged):

        # General Features
        merged["total_uv"] = merged["ma_uv"]+ merged["pc_uv"] #merged["mw_uv"]
        
        merged["average_discount"] = merged["dc_amt"] / (merged['ord_qty']+1)
        merged["ord_occ_yn"] = (merged["ord_occ_yn"] == 'Y') * 1
        merged["price_rate"] = merged["mobil_dc_prc"] / merged["mobil_sellprc"]

        merged["hesitate"] = merged.ord_qty / ( merged.cart_cnt+1)
        
        # Time-lagged Features (3days)
        merged["total_uv_3days"] = merged.total_uv[::-1].rolling(3).mean()[::-1]
        merged["sell_prc_3days"] = merged.mobil_sellprc[::-1].rolling(3).mean()[::-1]
        merged["dc_prc_3days"] = merged.mobil_dc_prc[::-1].rolling(3).mean()[::-1]
        merged["lwst_sell_prc_3days"] = merged.mobil_lwst_sellprc[::-1].rolling(3).mean()[::-1]
        merged["ord_amt_3days"] = merged.ord_amt[::-1].rolling(3).mean()[::-1]
        merged["dc_amt_3days"] = merged.dc_amt[::-1].rolling(3).mean()[::-1]
        merged["rlord_amt_3days"] = merged.rlord_amt[::-1].rolling(3).mean()[::-1]
        merged["prft_amt_3days"] = merged.prft_amt[::-1].rolling(3).mean()[::-1]
        merged["ord_mbr_cnt_3days"] = merged.ord_mbr_cnt[::-1].rolling(3).mean()[::-1]
        merged["ord_qty_3days"] = merged.ord_qty[::-1].rolling(3).mean()[::-1]
        merged["cart_cnt_3days"] = merged.cart_cnt[::-1].rolling(3).mean()[::-1]
        merged["ord_occ_yn_3days"] = merged.ord_occ_yn[::-1].rolling(3).mean()[::-1]
        merged["average_discount_3days"] =  merged.average_discount[::-1].rolling(3).mean()[::-1]
        merged["price_rate_3days"] = merged.price_rate[::-1].rolling(3).mean()[::-1]

        # Time-lagged Features (7days)
        merged["total_uv_7days"] = merged.total_uv[::-1].rolling(7).mean()[::-1]
        merged["sell_prc_7days"] = merged.mobil_sellprc[::-1].rolling(7).mean()[::-1]
        merged["dc_prc_7days"] = merged.mobil_dc_prc[::-1].rolling(7).mean()[::-1]
        merged["lwst_sell_prc_7days"] = merged.mobil_lwst_sellprc[::-1].rolling(7).mean()[::-1]
        merged["ord_amt_7days"] = merged.ord_amt[::-1].rolling(7).mean()[::-1]
        merged["dc_amt_7days"] = merged.dc_amt[::-1].rolling(7).mean()[::-1]
        merged["rlord_amt_7days"] = merged.rlord_amt[::-1].rolling(7).mean()[::-1]
        merged["prft_amt_7days"] = merged.prft_amt[::-1].rolling(7).mean()[::-1]
        merged["ord_mbr_cnt_7days"] = merged.ord_mbr_cnt[::-1].rolling(7).mean()[::-1]
        merged["ord_qty_7days"] = merged.ord_qty[::-1].rolling(7).mean()[::-1]
        merged["cart_cnt_7days"] = merged.cart_cnt[::-1].rolling(7).mean()[::-1]
        merged["ord_occ_yn_7days"] = merged.ord_occ_yn[::-1].rolling(7).mean()[::-1]
        merged["average_discount_7days"] =  merged.average_discount[::-1].rolling(7).mean()[::-1]
        merged["price_rate_7days"] = merged.price_rate[::-1].rolling(7).mean()[::-1]
        
        # Customer Features
        merged['recom_reg_cnt_7days'] =   merged.recom_reg_cnt[::-1].rolling(7).sum()[::-1]

        """
        counter_7days = (merged.recom_avgscr != 0).rolling(7).mean()[::-1]
        counter_30days = (merged.recom_avgscr != 0).rolling(30).mean()[::-1]

        merged['recom_avgscr_7days'] =   (merged.recom_avgscr[::-1].rolling(7).mean()[::-1] / counter_7days).df.replace([np.inf, -np.inf], 0)
        merged['recom_avgscr_30days'] =   (merged.recom_avgscr[::-1].rolling(30).mean()[::-1] / counter_30days).fillna(0)
        """

        merged = merged.iloc[:31]

        merged.drop(["pc_uv", "recom_reg_cnt", "recom_avgscr"], 1, inplace=True)

        #assert merged.isna().sum().sum() == 0

        return merged

    def reward_engineering(self, merged):
        merged["rcr"] = (
            merged["prft_amt"] / (merged["total_uv"]+1)
        )
        merged["drcr"] = np.nan
        merged["drcr"].iloc[:-1] = (
            merged["rcr"].iloc[:-1].values - merged["rcr"].iloc[1:].values
        )
        merged.drop("rcr", 1, inplace=True)
        
        return merged

    def log_transform(self, merged):
        """
        sales_features = ['best_prc', 'sell_prc', 'sell_prc_3days', 'sell_prc_7days', 'discount']
        prft_features = ['prft_amt', 'prft_amt_3days', 'prft_amt_7days']
        rlord_features = ['rlord_amt', 'rlord_amt_3days', 'rlord_amt_7days']
        """
        no_transform_features = ['drcr',
                                 'ord_occ_yn','ord_occ_yn_3days','ord_occ_yn_7days','price_rate', 'price_rate_3days',\
                                 'price_rate_7days']
        
        log_modulus = lambda x : np.sign(x) * np.log1p(np.abs(x))

        merged[merged.columns[~merged.columns.isin(no_transform_features)]] = \
            log_modulus(merged[merged.columns[~merged.columns.isin(no_transform_features)]])

        return merged

    def save(self):
        self.merged.to_csv("./data/raw/" + self.itemid + ".csv", index=False)
        self.configuration.to_csv("./data/config/" + self.itemid + ".csv", index=False)




if __name__ == "__main__":

    if not os.path.exists("./data"):
        os.makedirs("./data")
        
    if not os.path.exists("./data/raw"):
        os.makedirs("./data/raw")

    if not os.path.exists("./data/config"):
        os.makedirs("./data/config")
        
    if not os.path.exists("./data/preprocessed"):
        os.makedirs("./data/preprocessed")
        
    if not os.path.exists("./data/category"):
        os.makedirs("./data/category")

        
    # Preprocessing Phase
    
    ca = call_data()
    df = ca.from_hdfs(
        "/moneymall/CouponFeature/tmp/preprocessing/pre_df", daily_data=True
    )
    ctg = pd.read_csv('./src/preprocessing/mcls.csv', dtype='object').std_ctg_mcls_id.values.tolist()

    df["dt"] = pd.to_datetime(df["dt"])
    df = df[df.dt.isin(pd.date_range(start="2020-02-04", end="2020-03-15"))] # 40 days before

    ctg_candidates = ca.from_table("SELECT item_id FROM SCOM_DW.dbo.item WHERE std_ctg_mcls_id IN (" + "'"+"','".join(ctg)+"')")
    ctg_candidates = ctg_candidates.values.squeeze().tolist()

    df = df[df.item_id.isin(ctg_candidates)]
    
    cnt_candidates = df.groupby('item_id')['cart_cnt'].sum()
    
    THRESHOLD = 5000

    for i in range(THRESHOLD):
        if THRESHOLD >= len(list(cnt_candidates[cnt_candidates>i].index)):
            cnt_candidates = list(cnt_candidates[cnt_candidates>(i-1)].index)
            break
        
    df = df[df.item_id.isin(cnt_candidates)]

    price_features = [
         'critn_dt',
         'item_id',
         'mobil_sellprc',
         'mobil_dc_prc',
         'mobil_lwst_sellprc',
         'ord_occ_yn',
         'ord_qty',
         'ord_amt',
         'dc_amt',
         'rlord_amt',
         'prft_amt',
         'ord_mbr_cnt',
         'recom_reg_cnt',
         'recom_avgscr',
         'std_ctg_mcls_id',
         'std_ctg_scls_id',
         'std_ctg_dcls_id',
        ]

    price_df = ca.from_table(
        f"SELECT " + ", ".join(price_features)
        + " FROM SCOM_USERSET..EADD_ITEM_PRC_OPTI_VAR WHERE ITEM_ID IN ('"
        + "','".join(df.item_id.unique().tolist())
        + "')")

    price_df = price_df.drop_duplicates()
    item_list = df.item_id.unique().tolist()

    def preprocess_item(itemid):
        try:
            data = Preprocessor(itemid)
            if len(data.merged) == 31:
                if data.merged.isna().sum().sum() == 1:
                    data.save()
        except:
            pass
            
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    pool.map(preprocess_item, item_list)
    pool.close()
    pool.join()

    
    #Category Aggregating Phase
    paths = [path for path in Path('./data/config').glob('*.csv')]

    categories = []
    for path in paths:
        categories.append(pd.read_csv(path).ctg_scls_id.values[0])

    prices = ca.from_table(
        f"SELECT AVG(RLORD_AMT) AS 'RLORD_AMT_CTG',AVG(DC_AMT) AS 'DC_AMT_CTG',"
        + "AVG(MOBIL_SELLPRC) AS 'MOBIL_SELLPRC_CTG', AVG(PRFT_AMT) AS 'PRFT_AMT_CTG',"
        + "AVG(ORD_AMT) AS 'ORD_AMT_CTG', CRITN_DT, STD_CTG_SCLS_ID" 
        + " FROM SCOM_USERSET..EADD_ITEM_PRC_OPTI_VAR "
        + "WHERE CRITN_DT>'2020-02-01' AND CRITN_DT<='2020-03-15'"
        + "GROUP BY CRITN_DT,  STD_CTG_SCLS_ID")
    
    log_modulus = lambda x : np.sign(x) * np.log1p(np.abs(x))
    
    ctg_features = ['rlord_amt_ctg', 'dc_amt_ctg', 'mobil_sellprc_ctg', 'prft_amt_ctg',\
                   'ord_amt_ctg']
    
    prices['critn_dt'] =pd.to_datetime(prices['critn_dt'])
    #Category CONCAT

        
    # Training & Validation Split Phase
    paths = [path for path in Path('./data/raw').glob('*.csv')]

    t_s = []
    t_ns = []
    t_r = []
    t_a = []

    v_s = []
    v_ns = []
    v_r = []
    v_a = []

    for path in tqdm.tqdm(paths):
        data = pd.read_csv(path)
        ctg = pd.read_csv('./data/config/'+path.name, dtype='object').ctg_scls_id.values[0]
        
        
        data['critn_dt'] = pd.date_range(start='2020-02-14',end='2020-03-15')[::-1]
        data = data.merge(prices[prices.std_ctg_scls_id == ctg], how = 'left')
        data.fillna(0, inplace=True)
        data = pd.concat((data,pd.get_dummies(data.critn_dt.dt.dayofweek,prefix='dow').iloc[:, :-1]), 1)
        data.drop(['critn_dt','std_ctg_scls_id'],1,inplace=True)
        
        data[data.columns[data.columns.isin(ctg_features)]] = \
            log_modulus(data[data.columns[data.columns.isin(ctg_features)]])
        
        t_s.append(data.drop(['price_rate', 'drcr'], 1).iloc[2:])
        t_a.append(data[['price_rate']].iloc[1:-1])
        t_ns.append(data.drop(['price_rate', 'drcr'], 1).iloc[1:-1])
        t_r.append(data[['drcr']][1:-1])
        
        v_s.append(data.drop(['price_rate', 'drcr'], 1).iloc[1])
        v_a.append(data[['price_rate']].iloc[0])
        v_ns.append(data.drop(['price_rate', 'drcr'], 1).iloc[0])
        v_r.append(data[['drcr']].iloc[0])


    t_s = pd.concat(t_s,axis=0)
    t_ns = pd.concat(t_ns,axis=0)
    t_r = pd.concat(t_r,axis=0)
    t_a = pd.concat(t_a,axis=0)

    v_s = pd.concat(v_s,axis=1).transpose()
    v_ns = pd.concat(v_ns,axis=1).transpose()
    v_r = pd.concat(v_r,axis=1).transpose()
    v_a = pd.concat(v_a,axis=1).transpose()

  
    dow_t_s = t_s[['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5']]
    t_s = t_s.drop(['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5'], 1)
    
    dow_t_ns = t_ns[['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5']]
    t_ns = t_ns.drop(['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5'], 1)
   
    
    mu_s = t_s.mean()
    sigma_s = t_s.std()

    t_s  = ((t_s - mu_s) / sigma_s).clip(-10, 10)
    t_ns  = ((t_ns - mu_s) / sigma_s).clip(-10, 10)
    t_r = np.sign(t_r) * np.log1p(np.abs(t_r))

    t_s = pd.concat((t_s, dow_t_s), 1)
    t_ns = pd.concat((t_ns, dow_t_ns), 1)
   


    dow_v_s = v_s[['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5']]
    v_s = v_s.drop(['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5'], 1)
    
    dow_v_ns = v_ns[['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5']]
    v_ns = v_ns.drop(['dow_0', 'dow_1', 'dow_2', 'dow_3','dow_4','dow_5'], 1)
    
    std_r = t_r.std()
    t_r = t_r/std_r
    t_r = t_r.clip(-10, 10)

    v_s  = ((v_s - mu_s) / sigma_s).clip(-10, 10)
    v_ns  = ((v_ns - mu_s) / sigma_s).clip(-10, 10)
    v_r = np.sign(v_r) * np.log1p(np.abs(v_r))
    
    v_r = v_r/std_r
    v_r = v_r.clip(-10, 10)
    
    v_s = pd.concat((v_s, dow_v_s), 1)
    v_ns = pd.concat((v_ns, dow_v_ns), 1)
    
    t_s.to_csv("./data/preprocessed/t_s.csv",index=False)
    t_a.to_csv("./data/preprocessed/t_a.csv",index=False)
    t_r.to_csv("./data/preprocessed/t_r.csv",index=False)
    t_ns.to_csv("./data/preprocessed/t_ns.csv",index=False)


    v_s.to_csv("./data/preprocessed/v_s.csv",index=False)
    v_a.to_csv("./data/preprocessed/v_a.csv",index=False)
    v_r.to_csv("./data/preprocessed/v_r.csv",index=False)
    v_ns.to_csv("./data/preprocessed/v_ns.csv",index=False)
    
  