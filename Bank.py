
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('F:\\bank-additional-full.csv',sep=';')


# In[2]:


data.shape


# In[3]:


tot=len(set(data.index))
last=data.shape[0]-tot
last


# In[4]:


data.isnull().sum()


# In[5]:


print(data.y.value_counts())
sns.countplot(x='y', data=data)
plt.show()


# In[6]:


cat=data.select_dtypes(include=['object']).columns
cat


# In[7]:


for c in cat:
    print(c)
    print("-"*50)
    print(data[c].value_counts())
    print("-"*50)
    


# In[8]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
data['y']=le.fit_transform(data['y'])


# In[9]:


data.drop('poutcome',axis=1,inplace=True)


# In[10]:


print( data['age'].quantile(q = 0.75) + 
                      1.5*(data['age'].quantile(q = 0.75) - data['age'].quantile(q = 0.25)))


# In[11]:


data['age']=data[data['age']<69.6]
data['age'].fillna(int(data['age'].mean()),inplace=True)


# In[12]:


data['age'].values


# In[13]:


data[['age','y']].groupby(['age'],as_index=False).mean().sort_values(by='y', ascending=False)


# In[14]:


# for x in data:
#     x['Sex'] = x['Sex'].map( {'female': 1, 'male': 0}).astype(int)


# In[15]:


data['age_slice'] = pd.cut(data['age'],5)
data[['age_slice', 'y']].groupby(['age_slice'], as_index=False).mean().sort_values(by='age_slice', ascending=True)


# In[16]:


data['age'] = data['age'].astype(int)
data.loc[(data['age'] >= 16) & (data['age'] <= 28), 'age'] = 1
data.loc[(data['age'] > 28) & (data['age'] <= 38), 'age'] = 2
data.loc[(data['age'] > 38) & (data['age'] <= 49), 'age'] = 3
data.loc[ (data['age'] > 49) & (data['age'] <= 59), 'age'] = 4
data.loc[ (data['age'] > 59 )& (data['age'] <= 69), 'age'] = 5


# In[17]:


data.drop('age_slice',axis=1,inplace=True)


# In[18]:


data['marital'].replace(['divorced' ,'married' , 'unknown' , 'single'] ,['single','married','unknown','single'], inplace=True)


# In[19]:


data['marital']=le.fit_transform(data['marital'])


# In[20]:


data


# In[21]:


data['job'].replace(['student'] ,['unemployed'], inplace=True)


# In[22]:


data[['education', 'y']].groupby(['education'], as_index=False).mean().sort_values(by='education', ascending=True)


# In[23]:


fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sns.countplot(x = 'education', hue = 'loan', data = data)
ax.set_xlabel('Education', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_title('Education Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()


# In[24]:


fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sns.countplot(x = 'job', hue = 'loan', data = data)
ax.set_xlabel('job', fontsize=17)
ax.set_ylabel('y', fontsize=17)
ax.set_title('Education Count Distribution', fontsize=17)
ax.tick_params(labelsize=17)
sns.despine()


# In[25]:


data['education'].replace(['basic.4y','basic.6y','basic.9y','professional.course'] ,['not_reach_highschool','not_reach_highschool','not_reach_highschool','university.degree'], inplace=True)


# In[26]:


ohe=OneHotEncoder()
data['default']=le.fit_transform(data['default'])
data['housing']=le.fit_transform(data['housing'])
data['loan']=le.fit_transform(data['loan'])
data['month']=le.fit_transform(data['month'])
ohe=OneHotEncoder(categorical_features=data['month'])
data['contact']=le.fit_transform(data['contact'])
data['day_of_week']=le.fit_transform(data['day_of_week'])
data['job']=le.fit_transform(data['job'])
data['education']=le.fit_transform(data['education'])


# In[27]:


cat=data.select_dtypes(include=['object']).columns
cat


# In[28]:


def outlier_detect(data,feature):
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    lower  = q1-1.5*iqr
    upper = q3+1.5*iqr
    data = data.loc[(data[feature] > lower) & (data[feature] < upper)]
    print('lower IQR and upper IQR of',feature,"are:", lower, 'and', upper, 'respectively')
    return data


# In[29]:


data.columns


# In[30]:


data['pdays'].unique()


# In[31]:


data['pdays'].replace([999] ,[0], inplace=True)


# In[32]:


data['previous'].unique()


# In[33]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'campaign',  palette="rocket", data = data)
ax.set_xlabel('campaign', fontsize=25)
ax.set_ylabel('y', fontsize=25)
ax.set_title('campaign', fontsize=25)
sns.despine()


# In[34]:


sns.countplot(x = 'pdays',  palette="rocket", data = data)
ax.set_xlabel('pdays', fontsize=25)
ax.set_ylabel('y', fontsize=25)
ax.set_title('pdays', fontsize=25)
sns.despine()


# In[35]:


data[['pdays', 'y']].groupby(['pdays'], as_index=False).mean().sort_values(by='pdays', ascending=True)


# In[36]:


sns.countplot(x = 'emp.var.rate',  palette="rocket", data = data)
ax.set_xlabel('emp.var.rate', fontsize=25)
ax.set_ylabel('y', fontsize=25)
ax.set_title('emp.var.rate', fontsize=25)
sns.despine()


# In[37]:


outlier_detect(data,'duration')
#outlier_detect(data,'emp.var.rate')
outlier_detect(data,'nr.employed')
#outlier_detect(data,'euribor3m')


# In[38]:


X = data.iloc[:,:-1]
X = X.values
y = data['y'].values


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


algo = {'LR': LogisticRegression(), 
        'DT':DecisionTreeClassifier(), 
        'RFC':RandomForestClassifier(n_estimators=100), 
        'SVM':SVC(gamma=0.01),
        'KNN':KNeighborsClassifier(n_neighbors=10)
       }

for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))

