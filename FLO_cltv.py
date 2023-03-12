#############################################
# BG-NBD ve Gamma-Gamma ile CLTV Tahmini
#############################################

#############################################
# İş Problemi / Business Problem
#############################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları
# potansiyel değerin tahmin edilmesi gerekmektedir.

#############################################
# Veri Seti Hikayesi / Dataset Story
#############################################
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır./ The dataset consists of the information obtained from the past shopping behaviors of customers 
# who made their last purchases from Flo as OmniChannel (both online and offline shopping access) in the years 2020-2021
# 12 Değişken 19.945 Gözlem 2.7MB
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
# Adım 1: flo_data_20K.csv verisini okuyunuz.

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x) # Virgülden sonra 4 basamak göster ayarı yapıldı
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_ = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/flo_data_20k.csv")
df = df_.copy()
df.info()
df.describe().T


# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir. Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

# Aşağıdaki 0.01 ve 0.99 değerleri veri setimizde 10000000 gibi veri setimizi içerisinde olmasını beklemediğimiz değerler olması dahilinde onları baskılıyoruz.
#Fazla baskılamamak adına da 25 e 75 lik oran aralığı fazla daralttığı için, ucundan bakıp çıkmak adına 0.001 ve 0.99 değerleri seçiliyor
##Aşağıdaki fonksiyonun görevi kendisine girilen dğişken için eşik değer belirlemektir. Bundan dolayı öncelikle bir eşik
#değer belirlemek gerekmektedir. Bu işlem için;
 #-öncelikle çeyrek değerleri(25'lik(1.çyrk) ve 75'lik(3.çyrk) çeyrek değerler) hesaplayacağız
 #-çeyrek değerlerin farkı hesaplandıktan sonra
 #-Üst sınır için = Yukarıda elde ettiğimiz fark(interquantile range) 1.55 ile çarpılır ve 3. çeyrek değere eklenir.
 #-Alt limit için = 1. çeyrek değerden, yukarıda elde ettiğimiz fark(interquantile range) 1.55 ile çarpılarak çıkartılır.

# -Quantile fonksiyonu çeyreklik hesaplamak için kullanılır. Çeyreklik hesaplamak demek; değişkeni küçükten büyüğe sırala,
#yüzdelik olarak %25. , %50. değerlere karşılık gelenleri bulduğumuzda bunlar değişkenimizin çeyrek değerleridir.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


## Aşağıdaki fonksiyonumuzda bir dataframe ile değişken olarak çağırdığımızda bir değişkenin değeri üst eşik değerinden yüksek
# ise bu değeri üst eşik değeri ile değiştir diyoruz. Aynı şekilde alt eşik değerimizden küçük ise değer onu da alt eşik 
# değeri ile değiştir diyoruz fonksiyonumuz ile. Yani aykırı değerleri baskılamış oluyoruz. 

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)



# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayınız.

a = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]

for col in a:
    replace_with_thresholds(df, col)

df.describe().T

# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["omnichannel_total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_total_price_num"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


#for col in df.columns:
   # if "date" in col:
       # df[col] = pd.to_datetime(df[col])

df.info()
df.describe().T
###############################################################
# GÖREV 2. CLTV Veri Yapısının Oluşturulması
###############################################################
# Adım 1: Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir
# cltv dataframe'i oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure(T) değerleri ise
# haftalık cinsten ifade edilecek.
# recency: Müşterinin son satın alma tarihi - ilk satın alma tarihi
# T: analiz tarihinden(todaydate) ne kadar süre önce ilk satın alma yapılmış
# Frequency: tekrar eden toplam satın alma sayısı (frequency>1)(Müşteri en az 2 kez alışveriş yapmış)
# Monetary: satın alma başına ortalama kazanç (Monetary değerinin ortalaması) (toplam satınalma / toplam işlem !!!!!!!)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

# Alternatif yöntem; 
#Flo_cltv = {'customerId': df["master_id"],
           # 'recency_cltv_weekly': ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7,
            #'T_weekly': ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
            #'frequency': df["omnichannel_total_order_num"],
            #'monetary_cltv_avg': df["omnichannel_total_price_num"] / df["omnichannel_total_order_num"]}

#Flo_cltv = pd.DataFrame(Flo_cltv)

###############################################################
# GÖREV 3. BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###############################################################
# Adım 1: BG/NBD modelini fit ediniz

# BetaGeofitter methodumuz der ki bir model nesnesi oluşturacağım. Bu model nesnesi aracılığı ile sen fit methodunu
# kullanarak bana recency, frequency ve müşteri yaşı değerlerini verdiğinde sana bu modeli kurmuş olacağım der.
# Ve parametre bulma işlemleri sırasında bir argüman giriyoruz.
# panalizer_coef=0.001 giriliyor. Bu da katsayılara uygulanacak olan ceza katsayısıdır. Detayları makine öğrenmesinde karşımıza gelecek.


bgf = BetaGeoFitter(penalizer_coef=0.001)  # ceza puanı 0.001 i olabildiğince küçük tutuyoruz. Dataset ten datasete göre değişiklik gösterir
# bu penalizer coef değeri 0.001 ile 0.1 arasında değişiklik olarak gösterilebilir. Fit etmek modeli hazır hale getirmektir.


bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly']
        )


# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
# predict fonksiyonu BGNBG modeli için geçerli olup GamaGama modeli için geçerli değildir.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly']) # sıralı görtermek istersek sort_values(ascending=False) ekliyoruz.



cltv_df.head(20)
# Expected olanlar birim üzerinden
cltv_df["exp_sales_3_month"].describe().T

# 3 aylık periyotta şirketimizin beklediği satış sayısına erişmek istersek;
bgf.predict(4*3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


#Modelimiz ve gerçek değerler arasındaki durumu gözlemlemek amacı ile tablo oluşturmakta fayda var.

plot_period_transactions(Flo_bgf)
plt.show(block=True)


# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()
cltv_df.describe().T
cltv_df["expected_average_profit"].head(20)

# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#• Cltv değeri en yüksek 20 kişiyi gözlemleyiniz

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv.sort_values(by="cltv", ascending=False).head(20)

###############################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
###############################################################
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"]) # Küçük gördüğü yere D, büyük gördüklerine A
cltv_df.head()

cltv_df.groupby("cltv_segment").agg(["min", "max", "mean", "count"])
cltv_df.groupby("cltv_segment").agg({"exp_sales_3_month": ["min", "max", "mean", "count"]})


# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("cltv_segment").agg({"cltv": ["mean", "min", "max"]})

A_segments = cltv_df.loc[cltv_df["cltv_segment"] == "A"].sort_values(by="cltv" ,ascending=False)
B_segments = cltv_df.loc[cltv_df["cltv_segment"] == "B"].sort_values(by="cltv", ascending=False)

# Müşteri edinme maliyetlerini azaltmak adına üst segmentler olan A ve B segmentlerine odaklanarak A ve B segmentlerine özel
#ürün grupları oluşturulabilir.

#Tahminlerimizi dışa aktaralım. 
cltv_df.to_csv("Flo_cltv_prediction.csv")


###############################################################
# BONUS: Tüm süreci fonksiyonlaştırınız.
###############################################################

def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV veri yapısının oluşturulması
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)

cltv_df.head()
 
 
