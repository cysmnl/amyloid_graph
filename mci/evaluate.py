from subprocess import *
from pandas import Series, DataFrame
import os, re
import pandas as pd
import numpy as np

file='mci_atrophy_model_result.csv'
col = [ 'id','batch', 'epochs',
        'filters','poly_order','fc',
        'learn_rate','reg','dropout','momentum','decay_rate',
        'test_acc','sensitivity','ppv','specificity', 'npv']
folder_prefix='mci_atrophy_test'

#best settings
batch='5'
filters = '[5, 11]'
poly = '[4,4]'
fc = '128'
reg='0.1'
learn='0.001'

# create a master dataframe if it doesn't exist
if (os.path.isfile(file) == False):
    master = DataFrame(np.zeros((0,len(col))), columns=[col])
    master.to_csv(file,index=False)
    num=0
else:
    master=DataFrame(pd.read_csv(file), columns=[col])
    if len(master)==0:
        num=0
    else:
        num=int(re.search('[0-9]+', master.loc[len(master)-1,'id']).group()) + 1

for num in range(0,10):
    # create numpy array to store args
    args = ['python', 'mci.py']
    args = args + ['-m3']   #DEFINE
    args = args + ['-e']    #DEFINE
    args = args + ['--trainx', 'train_mci_x{}.csv'.format(num)]
    args = args + ['--trainy', 'train_mci_y{}.csv'.format(num)]
    args = args + ['--testx', 'test_mci_x{}.csv'.format(num)]
    args = args + ['--testy', 'test_mci_y{}.csv'.format(num)]
    
    master=DataFrame(pd.read_csv(file), columns=[col])

    # create data frame to store parameters for this op
    a = DataFrame(np.zeros((0,len(col))), columns=[col])

    # job id
    id_string=folder_prefix+str(num)
    args=args+['--dir_name', id_string]
    a.loc[0,'id']=id_string

    # batch size
    args=args+['--batch', batch]
    a.loc[0,'batch']=batch

    #epochs
    args=args+['--epochs', str(int(batch)*300)]

    # filters
    args=args+['--filters', filters]
    a.loc[0,'filters']=filters

    # polynomial order
    args=args+['--poly_order', poly]
    a.loc[0,'poly_order']=poly

    # fully connected layer
    args=args+['--fc',fc]
    a.loc[0,'fc']=fc

    # regularization
    args=args+['--reg',reg]
    a.loc[0,'reg']=reg

    # dropout
    args=args+['--dropout','1']
    a.loc[0,'dropout']=1

    # learning rate
    args=args+['--learn_rate',learn]
    a.loc[0,'learn_rate']=learn

    # decay rate
    args=args+['--decay_rate','1']
    a.loc[0,'decay_rate']=1

    # momentum
    args=args+['--momentum','0']
    a.loc[0,'momentum']=0

    # grab validation accuracy peak and mean
    print(args)
    output=check_output(args)
    output=output.decode()
    print(output)
    epoch=output[ (output.find('epoch',len(output)-240,len(output))+6) : (output.find('epoch',len(output)-240,len(output))+12) ]
    test_acc=output[ (output.find('test accuracy')+15) : (output.find('test accuracy')+20) ]
    sensitivity=output[ (output.find('sensitivity')+13) : (output.find('sensitivity')+17) ]
    ppv=output[ (output.find('ppv')+5) : (output.find('ppv')+9) ]
    specificity=output[ (output.find('specificity')+13) : (output.find('specificity')+17) ]
    npv=output[ (output.find('npv')+5) : (output.find('npv')+9) ]
    #print(peak, mean)

    # add values accordingly
    a.loc[0,'epochs'] = epoch
    a.loc[0,'test_acc'] = test_acc
    a.loc[0,'sensitivity'] = sensitivity
    a.loc[0,'ppv'] = ppv
    a.loc[0,'specificity'] = specificity
    a.loc[0,'npv'] = npv

    # finally add this to the master dataframe
    master=master.append(a)
    #print(master)

    # Export
    master.to_csv(file,index=False)
