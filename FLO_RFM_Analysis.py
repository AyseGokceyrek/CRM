#############################################
# RFM Analizi ile Müşteri Segmentasyonu/ customer Segmentation with RFM Analysis
#############################################

#############################################
# İş Problemi / Business Problem
#############################################
# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri
# belirlemek istiyor./ FLO, an online shoe store, wants to segment its customers and 
# determine marketing strategies according to these segments.
# 
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar
# oluşturulacak./ To this end, the behaviors of the customers will be defined and groups will be formed 
# according to the clusters in these behaviors.


#############################################
# Veri Seti Hikayesi / Dataset Story
#############################################
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışverişeriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır. / 
# The dataset consists of the information obtained from the past shopping behaviors of customers 
# who made their last purchases from Flo as OmniChannel (both online and offline shopping access) in the years 2020-2021.

# 12 Değişken(Variable) 19.945 Gözlem(values) 2.7MB
# Değişkenler / Variables
#
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


#############################################
# PROJE GÖREVLERİ / PROJECT TASKS
#############################################

#############################################
# GÖREV 1: Veriyi Anlama ve Hazırlama / Understanding and Preparing Data
#############################################

# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from datetime import datetime
df_ = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/flo_data_20k.csv")
df = df_.copy()
df.head()
df.shape

# Adım 2: Veri setinde;
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boş değer,
df.isnull().sum()
# e. Değişken tipleri incelemesi yapınız.
df.dtypes

# Yukarıda istenenleri tek bir fonksiyon ile de yapabiliriz.


def target_summary(dataframe, target, data_col):
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(data_col)[target].mean()}))


target_summary(df)

# Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["omnichannel_total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_total_price_num"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes

datetime = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[datetime] = df[datetime].apply(pd.to_datetime)

date_change = df.columns[df.columns.str.contains("date")]
df[date_change] = df[date_change].apply(pd.to_datetime)

# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

df["order_channel"].unique()
df["order_channel"].value_counts()

df.groupby("order_channel").agg({"master_id": "sum",
                                 "omnichannel_total_order_num": "sum",
                                 "omnichannel_total_price_num": "sum"})

# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"omnichannel_total_price_num": "sum"}).sort_values(by="omnichannel_total_price_num", ascending=False)

df.head()
df.info()

# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"omnichannel_total_order_num": "sum"}).sort_values(by="omnichannel_total_order_num", ascending=False)

# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.


def data_prepare(dataframe):
    dataframe["omnichannel_total_order_num"] = \
        dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["omnichannel_total_price_num"] = \
        dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    datetime = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[datetime] = df[datetime].apply(pd.to_datetime)

    return df

data_prepare(df)
df.dtypes

###############################################################
# GÖREV 2. RFM Metriklerinin Hesaplanması / Calculating RFM Metrics
###############################################################
# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

# Recency: Müşterimizin markamız ile en son iletişime geçmiş olduğu tarihi ifade etmektedir. Sıcaklık,Yenilik.
#   (Analizin yapıldığı tarih - ilgli müşterinin son satınalma yaptığı tarih)
# Frequency: Müşterinin yaptığı toplam satınalma ssayısı
# Monetary: Müşterinin yaptığı toplam satınalmalar neticesinde, müşterinin bırakmış olduğu parasal değerdir.



# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
df.head()

rfm = {'customerId': df["master_id"],
       'recency': (today_date - df["last_order_date"]).astype('timedelta64[D]'),
       'frequency': df["omnichannel_total_order_num"],
       'monetary': df["omnichannel_total_price_num"]}


# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

flo_rfm = pd.DataFrame(rfm)
type(flo_rfm[["recency"]])

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
# Not: recency değerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz

flo_rfm.describe().T

flo_rfm.shape
###############################################################
# GÖREV 3. RF Skorunun Hesaplanması
###############################################################
# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

flo_rfm["Recency_score"] = pd.qcut(flo_rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

flo_rfm["frequency_score"] = pd.qcut(flo_rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

flo_rfm["monetary_score"] = pd.qcut(flo_rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
flo_rfm.head()

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

flo_rfm["RFM_SCORE"] = (flo_rfm['Recency_score'].astype(str) +
                    flo_rfm['frequency_score'].astype(str))

###############################################################
# GÖREV 4. RF Skorunun Segment Olarak Tanımlanması
###############################################################
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz
# RFM isimlendirmesi
# seg_map = {
#     r'[1-2][1-2]': 'hibernating',
#     r'[1-2][3-4]': 'at_Risk',
#     r'[1-2]5': 'cant_loose',
#     r'3[1-2]': 'about_to_sleep',
#     r'33': 'need_attention',
#     r'[3-4][4-5]': 'loyal_customers',
#     r'41': 'promising',
#     r'51': 'new_customers',
#     r'[4-5][2-3]': 'potential_loyalists',
#     r'5[4-5]': 'champions'
# }

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


flo_rfm.head(20)
flo_rfm.tail()
flo_rfm['segment'] = flo_rfm['RFM_SCORE'].replace(seg_map, regex=True)

###############################################################
# GÖREV 5. Aksiyon Zamanı !
###############################################################
# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

flo_rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini
# csv olarak kaydediniz.

"""
a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
 tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
 iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
 yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
"""
df.columns
loyal_customer_ids = flo_rfm[flo_rfm["segment"].isin(["champions", "loyal_customers"])]["customerId"]
female_cust_ids = df[(df["master_id"].isin(loyal_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
female_cust_ids.to_csv("new_brand_target_customers.csv", index=False)
female_cust_ids.shape
len(female_cust_ids)
"""
b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
    -Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan 
        kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor. 
        Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
"""


tar_seg_cust_ids = flo_rfm[flo_rfm["segment"].isin(["cant_loose", "hibernating","new_customers"])]["customerId"]
tar_cust_ids = df[(df["master_id"].isin(tar_seg_cust_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
tar_cust_ids.to_csv("sale_target_customer_ids.csv", index=False)
tar_cust_ids.shape
len(tar_cust_ids)
tar_cust_ids.nunique()









