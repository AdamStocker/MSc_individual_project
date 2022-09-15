## Testing scripts 
import cdflib 
from time import time
from turtle import color
from jinja2 import pass_context
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from mpl_toolkits import mplot3d
from sunpy.net import Fido 
from sunpy.net.attrs import Instrument, Level, Time
from sunpy_soar.attrs import Identifier


def convertTime(EpochTime):
    name = []
    for j in EpochTime: 
        date_time_1 = datetime.datetime(2000,1,1,12,0) + datetime.timedelta(seconds=j*(10**(-9)))
        name.append(date_time_1)
    return name

## Dates Chosen for 1st iteration: Time("2020-07-12","2020-09-12")

instrument_mag = Instrument("MAG")
time_mag = Time("2021-08-20","2021-10-20")
level_mag = Level(2)
indentifier_mag = Identifier("MAG-RTN-BURST")

instrument_pas = Instrument("SWA")
time_pas = Time("2021-08-20","2021-10-20")
level_pas = Level(2)
indentifier_pas = Identifier("SWA-PAS-GRND-MOM")

result_mag = Fido.search(instrument_mag, time_mag, level_mag, indentifier_mag)
result_pas = Fido.search(instrument_pas,time_pas,level_pas,indentifier_pas)
files_mag = Fido.fetch(result_mag)
files_pas = Fido.fetch(result_pas)


mag_data = pd.DataFrame(data=[[0.0,0.0,0.0]],columns=["B_N","B_T","B_R"])
count = 0
for i in files_mag:
    cdf_file_MAG = cdflib.CDF(i) 
    try:
        epoch_time_MAG = cdf_file_MAG.varget("EPOCH")
        d_MAG = cdf_file_MAG.varget("B_RTN")
    except ValueError:
        epoch_time_MAG = [000000000000000000]
        d_MAG = pd.DataFrame(data=[[None,None,None]],columns=["B_N","B_T","B_R"])
    
    print(len(epoch_time_MAG))
    reduced_epoch_mag = []
    reduced_d_mag = []
    for index, value in enumerate(epoch_time_MAG):
        if index % 10 == 0:
            reduced_epoch_mag.append(value)

    for index, value in enumerate(d_MAG):
        if index % 10 == 0:
            reduced_d_mag.append(value)
        
    Mag_Date = convertTime(EpochTime=reduced_epoch_mag)

    S_mag = pd.DataFrame(reduced_d_mag,columns=["B_R","B_T","B_N"],index=Mag_Date)
    mag_data = pd.concat([mag_data,S_mag])

    count = count + 1 
    print(count)

print("found mag data")

pas_data = pd.Series(data=[0.0],name="pas data")
V_data = pd.DataFrame(data=[[0.0,0.0,0.0]],columns=["V_R","V_T","V_N"])
T_data = pd.Series(data=[0.0],name="Temp")

for i in files_pas:
    cdf_file_pas = cdflib.CDF(i)
    try:
        epoch_time_pas = cdf_file_pas.varget("EPOCH")
        T_pas = cdf_file_pas.varget("T")
        N_pas = cdf_file_pas.varget("N")
        V_RTN = cdf_file_pas.varget("V_RTN")
    except ValueError:
        epoch_time_pas = [000000000000000000]
        T_pas = pd.Series(data=[None],name="Temp")
        N_pas = pd.Series(data=[None],name="pas data")
        V_RTN = pd.DataFrame(data=[[None,None,None]],columns=["V_R","V_T","V_N"])    
    Pas_Date = convertTime(EpochTime=epoch_time_pas)
    
    S_pas = pd.Series(N_pas, index=Pas_Date,name="pas data")
    S_V_rtn = pd.DataFrame(V_RTN, index=Pas_Date,columns=["V_R","V_T","V_N"])
    T_pas = pd.Series(T_pas, index=Pas_Date,name="Temp")

    pas_data = pas_data.append(S_pas)
    V_data = pd.concat([V_data,S_V_rtn])
    T_data = T_data.append(T_pas)

print("all data collected")

mag_data = mag_data[1:]
pas_data = pas_data[1:] 
V_data = V_data[1:]
T_data = T_data[1:]

pas_data = pas_data.sort_index().dropna()
V_data = V_data.sort_index().dropna()
mag_data = mag_data.sort_index().dropna()
T_data = T_data.sort_index().dropna()

print("Data index sorted")

wanted_time = [900,1800,2700,3600]

data_res = (pas_data.index[-1]-pas_data.index[0]) / len(pas_data.index)
data_res = int(data_res.total_seconds())

for i in wanted_time:
    if i == 900:
        time_string = "900"
        number = "1/4"
    elif i == 1800:
        time_string = "1800"
        number = "2/4"
    elif i == 2700:
        time_string = "2700"
        number = "3/4"
    elif i == 3600:
        time_string = "3600"
        number = "4/4"
    else:
        print("Absolutely Stacked It")
    
    print("Working on "+number)
    num_data = int((i/data_res))

    ## NON-normalised mag data

    magnitude_mag_non_norm = np.sqrt(mag_data["B_R"]**2 + mag_data["B_T"]**2 + mag_data["B_N"]**2)
    #magnitude_mag_non_norm = pd.Series(magnitude_mag_non_norm)

    Mag_R = (mag_data["B_R"] - mag_data["B_R"].rolling(window=int(num_data)).mean()) / mag_data["B_R"].rolling(window=int(num_data)).mean()
    Mag_T = (mag_data["B_T"] - mag_data["B_T"].rolling(window=int(num_data)).mean()) / mag_data["B_T"].rolling(window=int(num_data)).mean()
    Mag_N = (mag_data["B_N"] - mag_data["B_N"].rolling(window=int(num_data)).mean()) / mag_data["B_N"].rolling(window=int(num_data)).mean()

    Mag_R.name = "Mag_R"
    Mag_T.name = "Mag_T"
    Mag_N.name = "Mag_N"
    magnitude_mag_non_norm.name = "magnitude_mag_non_norm"
  
    mag_field = pd.concat([Mag_R,Mag_T,Mag_N,magnitude_mag_non_norm],axis=1) 

    #mag_data["magnitude_mag"] = np.sqrt(mag_field["Mag_R"]**2 + mag_field["Mag_T"]**2 + mag_field["Mag_N"]**2)
  
    mag_data.reset_index(inplace=True)
    pas_data = pas_data.reset_index()

    # mag_data.sort_values(by="index")
    # pas_data.sort_values(by="index")

    df = pd.merge_asof(pas_data.sort_values("index"),mag_data.sort_values("index"),on=["index"],direction="nearest").dropna()#.drop_duplicates(subset="pas data",keep="first")
    df.set_index("index",inplace=True)
    mag_data.set_index("index",inplace=True)
    pas_data.set_index("index",inplace=True)
 
    df = pd.concat([df,V_data,T_data],axis=1)

    df = df.drop(["B_N","B_T","B_R"],axis=1)


    df["pas_normalised"] = (df["pas data"] - df["pas data"].rolling(window=int(num_data)).mean())/df["pas data"].rolling(window=int(num_data)).mean()
    mag_field["mag_normalised"] = (mag_field["magnitude_mag_non_norm"] - mag_field["magnitude_mag_non_norm"].rolling(window=int(num_data)).mean())/mag_field["magnitude_mag_non_norm"].rolling(window=int(num_data)).mean()

    df.dropna(inplace=True)


    #df["magnitude_V"] = np.sqrt(df["V_R"]**2 + df["V_T"]**2 + df["V_N"]**2)

    ##Finding V_0
    ## normalise first
    V_R_norm = (df["V_R"] - df["V_R"].rolling(window=int(num_data)).mean()) / df["V_R"].rolling(window=int(num_data)).mean()
    V_T_norm = (df["V_T"] - df["V_T"].rolling(window=int(num_data)).mean()) / df["V_T"].rolling(window=int(num_data)).mean()
    V_N_norm = (df["V_N"] - df["V_N"].rolling(window=int(num_data)).mean()) / df["V_N"].rolling(window=int(num_data)).mean()


    V_R_norm.name = "V_R_norm"
    V_T_norm.name = "V_T_norm"
    V_N_norm.name = "V_N_norm"

    df = pd.concat([df,V_R_norm,V_T_norm,V_N_norm],axis=1)
    
    #mag_field = pd.concat([Mag_R,Mag_T,Mag_N],axis=1)

    mag_field.reset_index(inplace=True)
    df.reset_index(inplace=True)
  
    df.dropna(inplace=True)
    mag_field.dropna(inplace=True)

    df = pd.merge_asof(df.sort_values("index"),mag_field.sort_values("index"),on=["index"],direction="nearest")#.dropna()#.drop_duplicates(subset=df,keep="first")
 
    df.set_index("index",inplace=True)
    
    print("After merge_asof"+number)
    B_0_R = df["Mag_R"].rolling(window=num_data).mean()[::num_data]
    B_0_T = df["Mag_T"].rolling(window=num_data).mean()[::num_data]
    B_0_N = df["Mag_N"].rolling(window=num_data).mean()[::num_data]

    B_0_R.name = "B_0_R"
    B_0_T.name = "B_0_T"
    B_0_N.name = "B_0_N"

    df = pd.concat([df,B_0_R,B_0_T,B_0_N],axis=1)

    #print(df)
    df["B_0_R"].fillna(method="bfill",inplace=True)
    df["B_0_T"].fillna(method="bfill",inplace=True)
    df["B_0_N"].fillna(method="bfill",inplace=True)

    df["vel_parr"] = df["V_R_norm"]*df["B_0_R"]+df["V_T_norm"]*df["B_0_T"]+df["V_N_norm"]*df["B_0_N"]
    df.dropna(inplace=True)
    df["vel_parr"] = df["vel_parr"]/(df["mag_normalised"].rolling(window=num_data).mean())
    df.dropna(inplace=True)
    
    print("finding correlation "+number)
    average_mag = df["mag_normalised"].rolling(window=int(num_data)).mean()[::int(num_data)]
    average_pas = df["pas_normalised"].rolling(window=int(num_data)).mean()[::int(num_data)]
    average = (average_mag + average_pas)/2

    correlation = df["pas_normalised"].rolling(int(num_data)).corr(df["mag_normalised"])[::int(num_data)]

    vel_par_average = df["vel_parr"].rolling(window=int(num_data)).mean()[::int(num_data)]
    vel_pas_corr = df["vel_parr"].rolling(int(num_data)).corr(df["pas_normalised"])[::int(num_data)]

    all_data = pd.DataFrame([correlation,average,vel_pas_corr], index=["correlation","average","vel_pas_corr"], columns=correlation.index).T

    all_data[all_data<-1000]=None

    all_data = all_data.dropna()

    all_data.to_csv("Cor_av_CvN_2021"+time_string)
    
    df = df[["pas data","mag_normalised","V_R","V_T","V_N","Temp"]]
    df.to_csv("data_frame2021_"+time_string)

