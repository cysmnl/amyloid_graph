#Feb-6
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy import nan as NA
from datetime import datetime
import os

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def av45(file):
    """Baseline PET AV-45 positivity estimates (labels)
    
    Args:
        file: File location to the csv file.
    
    Returns:
        A Pandas dataframe
    """
    labels = DataFrame(pd.read_csv(file), columns=['RID','EXAMDATE', 'VISCODE2', 
        'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF'])
    print('Unique AV45-PET RIDs: {} \n'.format(len(labels.RID.unique())))
    labels = labels.loc[labels.VISCODE2=='bl'].reset_index(drop=True)
    labels = labels.drop(['VISCODE2'], axis=1)
    labels = labels.rename(columns={'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF':'AV45'})
    return labels

def dx(df, file, max_days):
    """Append baseline diagnosis
    
    Args:
        df: Pandas dataframe of the AV45-PET subjects
        file: File location to the csv file.
        max_days: Maximum days from AV45-PET examdate
    
    Returns:
        A Pandas dataframe
    """
    df = df.append(DataFrame(columns=['DX']))    
    dx = DataFrame(pd.read_csv(file), 
        columns=['RID','EXAMDATE','DXCHANGE', 'DXCURREN', 'DXCONV'])
    nodx = []
    for i, RID in enumerate(df.RID):
        if RID > 5296:  # no ADNI3
            break
        petdate = df.EXAMDATE[i]
        frame = dx.loc[dx.RID==RID].sort_values(by='EXAMDATE').reset_index(drop=True)
        if frame.index.max() >= 0:
            for j, date in enumerate(frame.EXAMDATE):
                days = days_between(petdate, date)
                if days <= max_days:
                    if frame.DXCHANGE.notnull()[j] == True:
                        df.loc[i,'DX'] = frame.DXCHANGE[j]
                        break
                    elif frame.DXCURREN.notnutll()[j] == True:
                        df.loc[i,'DX'] = frame.DXCURREN[j]
                        break
                elif j == frame.index.max():
                    nodx.append(int(RID))
        else:
            print('DX info for RID {} not found \n'.format(int(RID)))
    df = df[df.DX.notnull()].reset_index(drop=True)
    print('No DX within {} days found for {} RIDs: {} \n'.format(max_days,len(nodx),nodx))
    hc = df.loc[df.DX.isin([1,7,9])]
    mci = df.loc[df.DX.isin([2,4,8])]
    ad = df.loc[df.DX.isin([3,5,6])]
    print('Healthy:{}, MCI:{}, AD:{} \n'.format(len(hc),len(mci),len(ad)))
    return df

def adas(df, file, max_days):
    df = df.append(DataFrame(columns=['ADAS']))
    adas = DataFrame(pd.read_csv(file), columns=['RID','USERDATE','TOTAL13'])
    noadas = []
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = adas.loc[adas.RID==RID].sort_values(by='USERDATE').reset_index(drop=True)
        if frame.index.max() >= 0:
            for j in frame.index:
                if frame.USERDATE.isnull()[j] == True:
                    print('error: no USERDATE for {}'.format(RID))
                    break
                else:
                    days = days_between(petdate, frame.USERDATE[j])
                    if days <= max_days:
                        df.loc[i,'ADAS'] = frame.TOTAL13[j]
                        break
                    elif j == frame.index.max():
                        noadas.append(int(RID))
        else:
            print('ADAS info for RID {} not found \n'.format(int(RID)))
    print('No ADAS within {} days found for {} RIDs: {} \n'.format(max_days,len(noadas),noadas))
    df = df[df.ADAS.notnull()].reset_index(drop=True)
    return df

def cdr(df, file, max_days):
    df = df.append(DataFrame(columns=['CDR']))
    cdr = DataFrame(pd.read_csv(file), columns=['RID','EXAMDATE','USERDATE','USERDATE2','CDGLOBAL'])
    over_maxdays = []
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = cdr.loc[cdr.RID==RID].sort_values(by=['EXAMDATE','USERDATE','USERDATE2']).reset_index(drop=True)
        if frame.index.max() >= 0:
            for j in frame.index:
                if frame.EXAMDATE.notnull()[j] == True:
                    date = frame.EXAMDATE[j]
                elif frame.USERDATE.notnull()[j] == True:
                    date = frame.USERDATE[j]
                elif frame.USERDATE2.notnull()[j] == True:
                    date = frame.USERDATE2[j]
                days = days_between(petdate, date)
                if days <= max_days:
                    df.loc[i,'CDR'] = frame.CDGLOBAL[j]
                    break
                elif j == frame.index.max():
                    over_maxdays.append(int(RID))
        else:
            print('CDR info for RID {} not found \n'.format(int(RID)))
    print('No CDR within {} days found for {} RIDs: {} \n'.format(max_days,len(over_maxdays),over_maxdays))
    df = df[df.CDR.notnull()].reset_index(drop=True)
    return df

def mmse(df, file, max_days):
    df = df.append(DataFrame(columns=['MMSCORE']))
    mmse = DataFrame(pd.read_csv(file, low_memory=False), columns=['RID','EXAMDATE','USERDATE','USERDATE2','MMSCORE'])
    over_maxdays = []
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = mmse.loc[mmse.RID==RID].sort_values(by=['EXAMDATE','USERDATE','USERDATE2']).reset_index(drop=True)
        if frame.index.max() >= 0:
            for j in frame.index:
                if frame.EXAMDATE.notnull()[j] == True:
                    date = frame.EXAMDATE[j]
                elif frame.USERDATE.notnull()[j] == True:
                    date = frame.USERDATE[j]
                elif frame.USERDATE2.notnull()[j] == True:
                    date = frame.USERDATE2[j]
                days = days_between(petdate, date)
                if days <= max_days:
                    df.loc[i,'MMSCORE'] = frame.MMSCORE[j]
                    break
                elif j == frame.index.max():
                    over_maxdays.append(int(RID))
        else:
            print('MMSE info for RID {} not found \n'.format(int(RID)))
    print('No MMSE within {} days found for {} RIDs: {} \n'.format(max_days,len(over_maxdays),over_maxdays))
    df = df[df.MMSCORE.notnull()].reset_index(drop=True)
    return df

def mem(df, file, max_days):
    df = df.append(DataFrame(columns=['ADNI_EF']))
    df = df.append(DataFrame(columns=['ADNI_MEM']))
    mem = DataFrame(pd.read_csv(file, low_memory=False), columns=['RID','EXAMDATE','USERDATE','USERDATE2','ADNI_EF','ADNI_MEM'])
    over_maxdays = []
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = mem.loc[mem.RID==RID].sort_values(by=['EXAMDATE','USERDATE','USERDATE2']).reset_index(drop=True)
        if frame.index.max() >= 0:
            for j in frame.index:
                if frame.EXAMDATE.notnull()[j] == True:
                    date = frame.EXAMDATE[j]
                elif frame.USERDATE.notnull()[j] == True:
                    date = frame.USERDATE[j]
                elif frame.USERDATE2.notnull()[j] == True:
                    date = frame.USERDATE2[j]
                days = days_between(petdate, date)
                if days <= max_days:
                    df.loc[i,'ADNI_EF'] = frame.ADNI_EF[j]
                    df.loc[i,'ADNI_MEM'] = frame.ADNI_MEM[j]
                    break
                elif j == frame.index.max():
                    over_maxdays.append(int(RID))
        else:
            print('MEM & EF info for RID {} not found \n'.format(int(RID)))
    print('No MEM & EF within {} days found for {} RIDs: {} \n'.format(max_days,len(over_maxdays),over_maxdays))
    df = df[df.ADNI_EF.notnull()].reset_index(drop=True)
    df = df[df.ADNI_MEM.notnull()].reset_index(drop=True)
    return df

def demo(df, file):
    df = df.append(DataFrame(columns=['AGE']))
    df = df.append(DataFrame(columns=['GEN']))
    df = df.append(DataFrame(columns=['EDU']))
    demo = DataFrame(pd.read_csv(file, low_memory=False), columns=['RID', 'PTGENDER','PTDOBMM','PTDOBYY','PTEDUCAT'])
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = demo.loc[demo.RID==RID].reset_index(drop=True)
        if frame.index.max() >= 0:        
            df.loc[i,'AGE'] = int(df.EXAMDATE[i][0:4]) - frame.PTDOBYY[0]
            df.loc[i,'GEN'] = frame.PTGENDER[0]
            df.loc[i,'EDU'] = frame.PTEDUCAT[0]
        else:
            print('Demographic info for RID {} not found \n'.format(int(RID)))
    df = df[df.AGE.notnull()].reset_index(drop=True)
    df = df[df.GEN.notnull()].reset_index(drop=True)
    df = df[df.EDU.notnull()].reset_index(drop=True)
    return df

def apoe(df, file):
    df = df.append(DataFrame(columns=['APOE']))
    apoe = DataFrame(pd.read_csv(file, low_memory=False), columns=['RID', 'APGEN1', 'APGEN2'])
    no_match = []
    for i, RID in enumerate(df.RID):
        petdate = df.EXAMDATE[i]
        frame = apoe.loc[apoe.RID==RID].reset_index(drop=True)
        if frame.index.max() >= 0: 
            df.loc[i,'APOE'] = 0
            if frame.APGEN1[0] == 4:
                df.loc[i,'APOE'] += 1
            if frame.APGEN2[0] == 4:
                df.loc[i,'APOE'] += 1            
        else:
            no_match.append(int(RID))
    print('Genetic info for {} RIDs not found: {} \n'
        .format(len(no_match),no_match))
    df = df[df.APOE.notnull()].reset_index(drop=True)
    return df

def vol(df, file1, file2, file3, max_days):
    # use roi region names to find the corresponding field names
    col = ['RID','EXAMDATE','IMAGETYPE','OVERALLQC','ST10CV']
    roi = DataFrame(pd.read_csv(file1), columns=['region'])
    name = DataFrame(pd.read_csv(file2), columns=['FLDNAME','TEXT'])

    # append field names as column names
    a = name.TEXT.str.contains("Volume")
    for i in roi.index:
        # print(roi.region[i])
        b = name.TEXT.str.contains(roi.region[i])
        for j in name.index:
            # print(vol.FLDNAME[j])
            if (a[j] == True and b[j] == True):
                col.append(name.FLDNAME[j])
    vol = DataFrame(pd.read_csv(file3),columns=col)
    
    # keep QC Pass and Non-Accelerated
    vol = vol.loc[vol.OVERALLQC == 'Pass'].loc[vol.IMAGETYPE == 'Non-Accelerated T1'].reset_index(drop=True)

    # normalize to ICV Cortical Volume, multiply by 10000
    num_col_labels = len(vol.iloc[0,:])
    for i in range(num_col_labels):
        if i >= (num_col_labels-86):
            vol.iloc[:,i] = (vol.iloc[:,i] / vol.loc[:,'ST10CV']) * 10000
    vol = vol.round(decimals=2)

    # append to df
    no_match = []
    df = df.append(DataFrame(columns=col[4:]))
    for i, RID in enumerate(df.RID):
        frame = vol.loc[vol.RID==RID].reset_index()
        if frame.index.max() >= 0:
            for j in frame.index:
                days = days_between(frame.EXAMDATE[j], df.EXAMDATE[i])
                if days <= max_days:
                    df.loc[i,'ST10CV':] = frame.loc[j,'ST10CV':]
                    break
        else:
            no_match.append(int(RID))
            pass
    print('No ROIs within {} days found for {} RIDs: {}'
        .format(max_days, len(no_match), no_match))        
    df = df[df.ST10CV.notnull()].reset_index(drop=True)
    return df

def zscore(df):
    # separate into diagnostic groups
    normal = data.loc[data.DX.isin([1,7,9])].reset_index(drop=True)
    mci = data.loc[data.DX.isin([2,4,8])].reset_index(drop=True)
    ad = data.loc[data.DX.isin([3,5,6])].reset_index(drop=True)
    print('HC:{}, MCI:{}, AD:{}'.format(len(normal),len(mci),len(ad)))

    # z-score
    super_normal = normal.loc[normal.APOE==0].loc[normal.AV45==0]
    ST = len(super_normal.iloc[0,:]) - 86
    super_normal = super_normal.iloc[:, ST:]
    mu = super_normal.mean()
    sigma = super_normal.std()


    #Normals
    zblock=(normal.iloc[:, ST:]-mu)/sigma
    normal.iloc[:,ST:]=zblock

    #MCI
    zblock=(mci.iloc[:, ST:]-mu)/sigma
    mci.iloc[:,ST:]=zblock

    #AD
    zblock=(ad.iloc[:, ST:]-mu)/sigma
    ad.iloc[:,ST:]=zblock

    return normal, mci, ad

def export(df, export=None):
    neg = df.loc[df.AV45==0].reset_index(drop=True)
    neg = neg.sample(n=len(neg)).reset_index(drop=True) # shuffle
    pos = df.loc[df.AV45==1].reset_index(drop=True)
    pos = pos.sample(n=len(pos)).reset_index(drop=True) # shuffle

    n = len(df)
    pneg = round(len(neg)/len(df),1)
    ppos = round(len(pos)/len(df),1)
    print('Total MCI: {}; {}% are AB+, {}% are AB-'.format(n, int(ppos*100), int(pneg*100) ))

    #how to split
    test = round(n*0.3)
    ntpos = round(test*ppos)
    ntneg = round(test*pneg)
    val = n//5
    nvpos = round(val*ppos)
    nvneg = round(val*pneg)
    train = n-val-test
    ntrain_pos = len(pos)-ntpos-nvpos
    ntrain_neg = len(neg)-ntneg-nvneg
    assert n == (ntpos + nvpos + ntrain_pos) + (ntneg + nvneg + ntrain_neg)

    #split
    mci_test = pos.iloc[:ntpos,:]
    mci_test = mci_test.append(neg.iloc[:ntneg,:])
    assert test == len(mci_test)
    mci_val = pos.iloc[ntpos:(ntpos+nvpos),:]
    mci_val = mci_val.append(neg.iloc[ntneg:(ntneg+nvneg),:])
    assert val == len(mci_val)
    mci_train = pos.iloc[(ntpos+nvpos):,:]
    mci_train = mci_train.append(neg.iloc[(ntneg+nvneg):,:])
    assert train == len(mci_train)

    # export = mci_test.sample(n=len(mci_test)).reset_index(drop=True) #shuffle
    # export.loc[:,'DX':].to_csv('../models/no_cerebellum/mci/test_mci_x.csv',index=False,header=False)
    # export.loc[:,'AV45_LABEL'].to_csv('../models/no_cerebellum/mci/test_mci_y.csv',index=False,header=False)

    # export = mci_val.sample(n=len(mci_val)).reset_index(drop=True) #shuffle
    # export.loc[:,'DX':].to_csv('../models/no_cerebellum/mci/val_mci_x.csv',index=False,header=False)
    # export.loc[:,'AV45_LABEL'].to_csv('../models/no_cerebellum/mci/val_mci_y.csv',index=False,header=False)

    # export = mci_train.sample(n=len(mci_train)).reset_index(drop=True) #shuffle
    # export.loc[:,'DX':].to_csv('../models/no_cerebellum/mci/train_mci_x.csv',index=False,header=False)
    # export.loc[:,'AV45_LABEL'].to_csv('../models/no_cerebellum/mci/train_mci_y.csv',index=False,header=False)

def generate(dx, num):
    # Generates random training and test sets
    cohort=dx
    dx=eval(dx)
    neg=dx.loc[dx.AV45_LABEL==0].reset_index(drop=True)
    neg=neg.sample(n=len(neg)).reset_index(drop=True) #shuffle
    pos=dx.loc[dx.AV45_LABEL==1].reset_index(drop=True)
    pos=pos.sample(n=len(pos)).reset_index(drop=True) #shuffle
    
    n=len(dx)
    pneg=round(len(neg)/len(dx),1);
    ppos=round(len(pos)/len(dx),1);
    print('Total {}: {}; {}% are AB+, {}% are AB-'.format(cohort, n, int(ppos*100), int(pneg*100) ))
    
    #how to split
    test=round(n*0.4)
    ntpos=round(test*ppos)
    ntneg=round(test*pneg)
    train=n-test
    ntrain_pos=len(pos)-ntpos
    ntrain_neg=len(neg)-ntneg
    assert n == (ntpos + ntrain_pos) + (ntneg + ntrain_neg)
    
    #split
    test_data = pos.iloc[:ntpos,:]
    test_data = test_data.append(neg.iloc[:ntneg,:])
    assert test == len(test_data)
    train_data = pos.iloc[ntpos:,:]
    train_data = train_data.append(neg.iloc[ntneg:,:])
    assert train == len(train_data)
    
    for i in range(num):
        print('Exporting', i)
        export1 = test_data.sample(n=len(test_data)).reset_index(drop=True) #shuffle
        export1.loc[:,'DX':].to_csv('../models/no_cerebellum/{}/test_{}_x{}.csv'.format(cohort, cohort, str(i)),index=False,header=False)
        export1.loc[:,'AV45_LABEL'].to_csv('../models/no_cerebellum/{}/test_{}_y{}.csv'.format(cohort, cohort, str(i)),index=False,header=False)

        export2 = train_data.sample(n=len(train_data)).reset_index(drop=True) #shuffle
        export2.loc[:,'DX':].to_csv('../models/no_cerebellum/{}/train_{}_x{}.csv'.format(cohort, cohort, str(i)),index=False,header=False)
        export2.loc[:,'AV45_LABEL'].to_csv('../models/no_cerebellum/{}/train_{}_y{}.csv'.format(cohort, cohort, str(i)),index=False,header=False)