import pandas as pd

df = pd.read_excel(r'C:/Users/tianyunchuan/Desktop/bd01.xlsx')
l_var, l_proc = [], []

for i in range(len(df)):
    l_clean = [x for x in list(df.iloc[i]) if type(x)!=float]
    for k in range(len(l_clean)):
        if k == 0:
            l_proc.append(l_clean[k])
            l_proc.append(r'{}-{}'.format('1', len(l_clean)-1))
        else:
            l_var.append(l_clean[k])
print(l_var)
print(l_proc)    
pd.DataFrame(l_var).T.to_excel(r'c:/temp/l_var.xlsx')
pd.DataFrame(l_proc).T.to_excel(r'c:/temp/l_proc.xlsx')