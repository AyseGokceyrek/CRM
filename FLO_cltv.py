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

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


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
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
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

Flo_cltv = {'customerId': df["master_id"],
            'recency_cltv_weekly': ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7,
            'T_weekly': ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
            'frequency': df["omnichannel_total_order_num"],
            'monetary_cltv_avg': df["omnichannel_total_price_num"] / df["omnichannel_total_order_num"]}

Flo_cltv = pd.DataFrame(Flo_cltv)
type(Flo_cltv[["frequency"]])

Flo_cltv.head()
###############################################################
# GÖREV 3. BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###############################################################
# Adım 1: BG/NBD modelini fit ediniz
# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

Flo_bgf = BetaGeoFitter(penalizer_coef=0.001)  # ceza puanı 0.001 i olabildiğince küçük tutuyoruz. Dataset ten datasete göre değişiklik gösterir
# bu penalizer coef değeri 0.001 ile 0.1 arasında değişiklik olarak gösterilebilir. Fit etmek modeli hazır hale getirmektir.

Flo_bgf.fit(Flo_cltv['frequency'],
        Flo_cltv['recency_cltv_weekly'],
        Flo_cltv['T_weekly'])

Flo_cltv["exp_sales_3_month"] = Flo_bgf.predict(4*3,
                                                Flo_cltv['frequency'],
                                                Flo_cltv['recency_cltv_weekly'],
                                               Flo_cltv['T_weekly']).sort_values(ascending=False)

Flo_cltv.head(20)
# Expected olanlar birim üzerinden
Flo_cltv["exp_sales_3_month"].describe().T

Flo_cltv["exp_sales_6_month"] = Flo_bgf.predict(4*6,
                                                Flo_cltv['frequency'],
                                                Flo_cltv['recency_cltv_weekly'],
                                                Flo_cltv['T_weekly'])
Flo_cltv.describe().T
Flo_cltv.columns


plot_period_transactions(Flo_bgf)
plt.show(block=True)

#Modelimiz ve gerçek değerler arasındaki durumu gözlemlemek amacı ile tablo oluşturmakta fayda var.

# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz

Flo_ggf = GammaGammaFitter(penalizer_coef=0.01)

Flo_ggf.fit(Flo_cltv['frequency'].astype(int), Flo_cltv['monetary_cltv_avg'])

Flo_ggf.conditional_expected_average_profit(Flo_cltv['frequency'],
                                        Flo_cltv['monetary_cltv_avg']).head(10)

Flo_ggf.conditional_expected_average_profit(Flo_cltv['frequency'],
                                        Flo_cltv['monetary_cltv_avg']).sort_values(ascending=False).head(10)

Flo_cltv["expected_average_profit"] = Flo_ggf.conditional_expected_average_profit(Flo_cltv['frequency'],
                                                                             Flo_cltv['monetary_cltv_avg'])



Flo_cltv.describe().T
Flo_cltv["expected_average_profit"].head(20)
# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#• Cltv değeri en yüksek 20 kişiyi gözlemleyiniz

Flo_cltv["cltv"] = Flo_ggf.customer_lifetime_value(Flo_bgf,
                                   Flo_cltv['frequency'],
                                   Flo_cltv['recency_cltv_weekly'],
                                   Flo_cltv['T_weekly'],
                                   Flo_cltv['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

Flo_cltv.sort_values(by="cltv", ascending=False).head(20)

###############################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
###############################################################
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

Flo_cltv["cltv_segment"] = pd.qcut(Flo_cltv["cltv"], 4, labels=["D", "C", "B", "A"])
Flo_cltv.head()

Flo_cltv.groupby("cltv_segment").agg(["min", "max", "mean", "count"])

# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

Flo_cltv.groupby("cltv_segment").agg({"cltv": ["mean", "min", "max"]})

A_segments = Flo_cltv.loc[Flo_cltv["cltv_segment"] == "A"]sort_values(by="cltv" ,ascending=False)
B_segments = Flo_cltv.loc[Flo_cltv["cltv_segment"] == "B"].sort_values(by="cltv", ascending=False)

# Müşteri edinme maliyetlerini azaltmak adına üst segmentler olan A ve B segmentlerine odaklanarak A ve B segmentlerine özel
#ürün grupları oluşturulabilir.



Flo_cltv.to_csv("Flo_cltv_prediction.csv")
