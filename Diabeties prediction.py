#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import random
pallete = ['Accent_r', 'Blues', 'BrBG', 'BrBG_r', 'BuPu', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
           'GnBu', 'GnBu_r', 'OrRd', 'Oranges', 'Paired', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdGy_r',
           'RdPu', 'Reds', 'autumn', 'cool', 'coolwarm', 'flag', 'flare', 'gist_rainbow', 'hot', 'magma',
           'mako', 'plasma', 'prism', 'rainbow', 'rocket', 'seismic', 'spring', 'summer', 'terrain', 
           'turbo', 'twilight']


# In[9]:


db=pd.read_csv(r'C:\Users\laksh\OneDrive\Desktop\diabetes.csv')


# In[10]:


db


# In[11]:


db.info()


# In[13]:


db.head(10)


# In[6]:


db.describe()


# In[7]:


db.isnull().sum()


# # Exporatory Data Analysis

# # Pie Chart

# In[91]:


sns.set(style='darkgrid')
labels=['Healthy','Diabetic']
sizes=db['Outcome'].value_counts(sort=True)
colors=["orange","green"]
explode=(0,0)
plt.figure(figsize=(8,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90,)
plt.title("PIE CHART OF HEALTHY VS DIABETIC IN THE DATASET",color='black')
plt.show()


# In[9]:


db.hist(figsize=(15,12));


# In[10]:


sns.scatterplot(data=db,legend='auto')


# In[11]:


sns.histplot(x="Pregnancies", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='green',legend=True)
plt.title("Healthy vs Diabetic by Pregnancies",fontsize=15)


# In[12]:


sns.histplot(x="Glucose", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='yellow',legend=True,)
plt.title("Healthy vs Diabetic by Glucose")


# In[13]:


sns.histplot(x="BloodPressure", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by BloodPressure")


# In[14]:


sns.histplot(x="SkinThickness", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by SkinThickness")


# In[15]:


sns.histplot(x="Insulin", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by Insulin")


# In[16]:


sns.histplot(x="BMI", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by BMI")


# In[17]:


sns.histplot(x="DiabetesPedigreeFunction", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by DiabetesPedigreePrediction")


# In[18]:


sns.histplot(x="Age", hue="Outcome", data=db, kde=True, palette=random.choice(pallete),color='blue',legend=True)
plt.title("Healthy vs Diabetic by Age")


# In[19]:


sns.residplot(x=db['Age'],y=db['Outcome'])


# In[20]:


sns.residplot(x=db['Age'],y=db['Outcome'])


# In[21]:


sns.pairplot(db,hue='Outcome',palette=random.choice(pallete))


# # Data Preprocessing

# ## CLEANING THE DATASET

# In[22]:


db.isnull().sum()


# In[23]:


db.describe()


# In[24]:


db.corr()


# In[25]:


corr=db.corr()
corr[corr>0.5]


# ## SKEW HANDLING

# ### Dataset before Skew Handling

# In[26]:


db.skew()


# ### Heatmap before skew handling

# In[27]:


plt.figure(figsize=(10,10))
s=db.corr()
plt.suptitle("HEAT MAP OF CORRELATION MATRIX",color='red',ha='center')
sns.heatmap(data=s,linecolor='white',linewidths=0.1,square=True,annot=True)


# In[28]:


from scipy.stats import skew
for col in db:
    print(col)
    print(skew(db[col]))
    plt.figure()
    sns.displot((db[col]),kde=True)
    plt.show()
    


# ### Dataset after Skew Handling

# Here we are handling the skewed data by using "sqrt" function which we need to import from scipy.stats

# In[29]:


skew(db['Insulin'])
plt.figure()
sns.displot(db['Insulin'],kde=True)
plt.show()


# In[30]:


db['Pregnancies']=np.sqrt(db['Pregnancies'])


# In[31]:


skew(db['Pregnancies'])
print(skew(db['Pregnancies']))
plt.figure()
sns.displot((db['Pregnancies']),kde=True)
plt.show()


# In[32]:


db['DiabetesPedigreeFunction']=np.sqrt(np.sqrt(db['DiabetesPedigreeFunction']))


# In[33]:


skew(db['DiabetesPedigreeFunction'])
print(skew(db['DiabetesPedigreeFunction']))
plt.figure()
sns.displot((db['DiabetesPedigreeFunction']),kde=True)
plt.show()


# In[34]:


db['Age']=np.sqrt(db['Age'])


# In[35]:


skew(db['Age'])
print(skew(db['Age']))
plt.figure()
sns.displot((db['Age']),kde=True)
plt.show()


# # Heat map of Dataset after Skew Handling

# In[36]:


plt.figure(figsize=(10,10))
s=db.corr()
plt.suptitle("HEAT MAP OF CORRELATION MATRIX",color='red',ha='center')
sns.heatmap(data=s,linecolor='white',linewidths=0.1,square=True,annot=True)


# In[37]:


db.groupby('Outcome').size()


# ## Scatter plots

# In[38]:


def ScatterPlot(x , y, axis):
    return sns.scatterplot(x = x, y = y,hue = 'Outcome',ax = axis,data = db)


# In[39]:


fig, ax = plt.subplots(3, 2, figsize=(15, 15))
ax1 = ax[0, 0]
ax2 = ax[0, 1]
ax3 = ax[1, 0]
ax4 = ax[1, 1]
ax5 = ax[2, 0]
ax6 = ax[2, 1]
ScatterPlot('Glucose','Age',ax1)
ScatterPlot('BloodPressure','Age',ax2)
ScatterPlot('SkinThickness','Age',ax3)
ScatterPlot('Insulin','Age',ax4)
ScatterPlot('BMI','Age',ax5)
ScatterPlot('DiabetesPedigreeFunction','Age',ax6)
plt.show()


# ## CHECKING OUTLIERS OF THE DATASET USING THE BOXPLOTS

# In[40]:


fig,axs=plt.subplots(4,2,figsize=(20,20))
axs=axs.flatten()
for i in range(len(db.columns)-1):
    sns.boxplot(data=db,x=db.columns[i],ax=axs[i],palette=random.choice(pallete))


# In[41]:


db.isnull().sum()


# In[42]:


db.iloc[:,:-1]


# ## Scaling the Dataset

# In[43]:


from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler()
mscalerdb=mmscaler.fit_transform(db)
mscalerdb


# ## Standardize the Dataset

# In[44]:


from sklearn.preprocessing import StandardScaler
sscaler=StandardScaler()
sscalerdb=sscaler.fit_transform(db.iloc[:,:-1])
sscalerdb


# ## Normalizing the Dataset

# In[45]:


from sklearn.preprocessing import Normalizer
nr=Normalizer()
nr1=nr.fit_transform(db)
nr1


# In[46]:


db.isnull().sum()


# # Classification

# ### Feature Selection

# In[92]:


x = db.iloc[:, :-1].values
y = db.iloc[:, -1].values


# ## SPLITTING DATA INTO TRAIN AND TEST SET

# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# ## RandomForestClassifier

# In[51]:


rf=RandomForestClassifier()
rf1= rf.fit(x_train,y_train)
yhat=rf1.predict(x_test)
acc_scores=dict()
acc_scores["RandomForestClassifier"]=accuracy_score(y_test,yhat)
print("Classification report--Random Forest Classifier")
print("****************************************************")
print(classification_report(y_test,yhat));
print("============================================")
print("                                            ")
titles_options = [("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rf1, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# ## GradientBoostingClassifier

# In[52]:


gcb=GradientBoostingClassifier()
grcb= gcb.fit(x_train,y_train)
yhat=grcb.predict(x_test)
acc_scores=dict()
acc_scores["GradientBoostingClassifier"]=accuracy_score(y_test,yhat)
print("Classification report--GradientBoostingClassifier")
print("****************************************************")
print(classification_report(y_test,yhat));
print("============================================")
print("                                            ")
titles_options = [("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(grcb, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# ## Support Vector Machine Classifier

# In[53]:


svc=SVC()
svc1= svc.fit(x_train,y_train)
yhat=svc1.predict(x_test)
acc_scores=dict()
acc_scores["SVM Classifier"]=accuracy_score(y_test,yhat)
print("Classification report--SVM Classifier")
print("****************************************************")
print(classification_report(y_test,yhat));
print("============================================")
print("                                            ")
titles_options = [("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svc1, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# ## DecisionTreeClassifier

# In[54]:


dt=DecisionTreeClassifier()
dt1=dt.fit(x_train,y_train)
yhat=dt1.predict(x_test)
acc_scores = dict()
acc_scores["DecisionTreeClassifier"] = accuracy_score(y_test,yhat)
print("Classification report - DecisionTreeClassifier")
print("============================================")
print(classification_report(y_test,yhat))
print("============================================")
print("                                            ")
titles_options = [("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(dt1, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# ## K-Nearest NeighbourClassifiers

# In[55]:


kn=KNeighborsClassifier()
knn=kn.fit(x_train,y_train)
yhat =knn.predict(x_test)
acc_scores=dict()
acc_scores["KNeighborsClassifier"]=accuracy_score(y_test,yhat)
print("Classification report - KNeighborsClassifier")
print("============================================")
print(classification_report(y_test,yhat))
print("============================================")
print("                                            ")
titles_options = [("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# ## Logistic Regression Model

# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
lm=LogisticRegression(max_iter=10000)
lm1=lm.fit(x_train,y_train)
yhat=lm1.predict(x_test)
acc_scores=dict()
acc_scores["Logistic regression"] = accuracy_score(y_test,yhat)
print("Classification report - Logistic regression ")
print("============================================")
print(classification_report(y_test,yhat))
print("============================================")
print("                                            ")
titles_options=[("Confusion matrix without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(lm1, x_test, y_test,display_labels=(0,1),cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    print("*******************************************")


# In[74]:


from sklearn.metrics import accuracy_score
plt.figure()
accuracy_score_df = pd.DataFrame()
accuracy_score_df["Model"] = acc_scores.keys()
accuracy_score_df["Accuracy"] = acc_scores.values()
ax = sns.barplot(x="Model",y="Accuracy",data=accuracy_score_df)
tx = ax.set_title("Comparion of accuracy Logistic regression vs Decision Tree vs Random Forest")
plt.show()


# In[87]:


import numpy as np
input_data=[2,197,70,45,543,30.5,0.158,53]
input_data_as_numpyarray=np.asarray(input_data)
input_datareshaped=input_data_as_numpyarray.reshape(1,-1)
print(input_datareshaped)
std_data=sscaler.transform(input_datareshaped)
print(std_data)
prediction=lm.predict(std_data)
print(prediction)


# In[105]:


print("Accuracy:",metrics.accuracy_score(y_test, yhat))
print("Precision:",metrics.precision_score(y_test, yhat))
print("Recall:",metrics.recall_score(y_test, yhat))


# In[103]:


from sklearn import metrics
y_pred_proba = lm.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print("AUC score is",auc)


# In[ ]:





# In[ ]:





# In[ ]:




