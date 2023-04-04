                 ######Telco Churn Feature Engineering######

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
#Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
#gerçekleştirmeniz beklenmektedir.

#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet
#hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden
#ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

#CustomerId = Müşteri İd’si
#Gender = Cinsiyet
#SeniorCitizen = Müşterinin yaşlı olup olmadığı (1, 0)
#Partner = Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
#Dependents = Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
#tenure = Müşterinin şirkette kaldığı ay sayısı
#PhoneService = Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
#MultipleLines = Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
#InternetService = Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
#OnlineSecurity = Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#OnlineBackup = Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#DeviceProtection = Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#TechSupport = Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
#StreamingTV = Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#StreamingMovies = Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#Contract = Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
#PaperlessBilling = Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
#PaymentMethod = Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
#MonthlyCharges = Müşteriden aylık olarak tahsil edilen tutar
#TotalCharges = Müşteriden tahsil edilen toplam tutar
#Churn = Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

                 ###############################################################
                    # Genel resmi incelemek için kütüphanelerimizi kuralım.
                 ###############################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', None)

df_ = pd.read_csv("6.Hafta/Feature Engineering/Telco-Customer-Churn.csv")
df = df_.copy()
df.head()
df.shape

def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)
                   #######################################################
                      #Numerik ve Kategorik değişkenleri yakalayalım.
                   #######################################################

df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].apply(lambda x:1 if x == "Yes" else 0)
df.head()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

                 ##############################################################
                     #Numerik ve Kategorik değişkenlerin analizini yapalım.
                 ##############################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

cat_summary(df, "Churn", plot=True)

def num_summary(dataframe,col_name,plot=False):
    quantiles = [.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99]
    print(dataframe[col_name].describe(quantiles).T)
    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

             ##############################################################################
                 # Hedef değişken analizi yapalım (Kategorik değişkenlere göre hedef
             #değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
             ##############################################################################

def target_summary_with_num(dataframe,target,num_cols):
    print(dataframe.groupby(target).agg({num_cols:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

def target_summary_with_cat(dataframe,target,cat_cols):
    print(cat_cols)
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(cat_cols)[target].mean(),
                       "Count":dataframe[cat_cols].value_counts(),
                       "Ratio":dataframe[cat_cols].value_counts()/len(dataframe)}), end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

                             #######################################
                                # Aykırı gözlem analizi yapalım.
                             #######################################

def outlier_thresholds(data, var, q1_range=.05, q3_range=.95):
    q1 = data[var].quantile(q1_range)
    q3 = data[var].quantile(q3_range)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr
    return low, up

def check_outlier(data, var):
    low, up = outlier_thresholds(data, var)
    return data[(data[var] < low) | (data[var] > up)].any(axis=None)

def replace_with_thresholds(dataframe, var):
    low, up = outlier_thresholds(dataframe, var)
    dataframe.loc[dataframe[var] < low, var] = low
    dataframe.loc[dataframe[var] > up, var] = up

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier:
        replace_with_thresholds(df, col)

                              #################################
                               # Eksik gözlem analizi yapalım.
                              #################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.groupby("Churn").agg({"TotalCharges": "median"})
df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("Churn")["TotalCharges"].transform("median"))
df.isnull().sum()
                         ###########################################################
                           # Korelasyon analizi yaparak bunu görselleştirelim.
                         ###########################################################

f, ax = plt.subplots(figsize=[13, 10])
sns.heatmap(df.corr(),annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

                                 ######################################
                                    # Yeni değişkenler oluşturalım.
                                 ######################################

df.loc[df["tenure"] <= 12, "NEW_TENURE"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE"] = "5-6 Year"

df["NEW_AVG_CHARGES"] = df["TotalCharges"]/(df["tenure"]+1)

df["NEW_INCREASE"] = df["NEW_AVG_CHARGES"]/df["MonthlyCharges"]

                           ###############################################
                              # Encoding işlemlerini gerçekleştirelim.
                           ###############################################

def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if (df[col].dtype not in ["int64","float64"]) and (df[col].nunique()==2)]
binary_cols

for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 10>=df[col].nunique()>2]
ohe_cols

def one_hot_encoder(dataframe,categorical_cols,drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df_final = one_hot_encoder(df,ohe_cols)
df_final.head()

                       ###########################################################
                          # Numerik değişkenler için standartlaştırma yapınız.
                       ###########################################################

scaler = StandardScaler()
df_final[num_cols] = scaler.fit_transform(df_final[num_cols])
df_final.head()
                                    ###########################
                                       # Model oluşturunuz.
                                    ###########################

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier (random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

#NOT: Diğer modelleme yöntemleriyle birlikte çok yakında yeniden analizler de gelecek:)
