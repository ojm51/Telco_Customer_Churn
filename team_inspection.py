import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # Notice graduation (눈금 표시)

url = r"withNaN_WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url, skiprows=0)

# dirty data
df.replace(" ", np.nan, inplace=True)  # null space -> nan
df.dropna(inplace=True)  # drop rows that contain nan value
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])  # data type : object -> float

# split target data
data = df.drop('Churn', axis=1)
target = df['Churn']

colors = ['#EAA130', '#57B0AA'] # set graph color ( orange & mint )

''' =========================================== gender distribution graph ================================================= '''
ax = (data['gender'].value_counts()*100.0 /len(data))\
    .plot.pie(autopct='%.1f%%', labels=['Male', 'Female'], figsize =(5, 5),  fontsize=12, colors=colors)
ax.set_ylabel(' ', fontsize=12)
ax.set_title('Gender Distribution', fontsize=12)
plt.show()

''' =========================================== SeniorCitizen distribution graph ================================================= '''
ax = (data['SeniorCitizen'].value_counts()*100.0 /len(data))\
.plot.pie(autopct='%.1f%%', labels = ['Young', 'Senior'],figsize =(5,5), fontsize = 12 ,colors=colors)
ax.set_ylabel('Senior Citizens',fontsize = 12)
ax.set_title('% of Senior Citizens', fontsize = 12)

plt.show()

''' =========================================== dependents(부양가족) ================================================= '''
ax = (data['Dependents'].value_counts()*100.0 /len(data))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 ,colors=colors)
ax.set_ylabel(' ',fontsize = 12)
ax.set_title('Dependents distribution', fontsize = 12)

plt.show()

''' =========================================== partner(결혼) ================================================= '''
ax = (data['Partner'].value_counts()*100.0 /len(data))\
.plot.pie(autopct='%.1f%%', labels = ['Single', 'Married'],figsize =(5,5), fontsize = 12 ,colors=colors)
ax.set_ylabel(' ',fontsize = 12)
ax.set_title('% of Married', fontsize = 12)

plt.show()

''' =========================================== dependents based on partner ================================================= '''
partner_dependents = data.groupby(['Partner','Dependents']).size().unstack()

ax = (partner_dependents.T*100.0 / partner_dependents.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (8,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # tick (눈금 표시)

# set legend property ( location : center, title : churn, size : 14 ) (범례 설정)
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)

ax.set_ylabel('% Customers',size = 14) # set ylabel name
ax.set_title('Distribution of Dependents based on Partner',size = 14) # set title
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart ( 막대 위에 숫자 표시 )
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)
plt.show()

''' =========================================== Various services graph ================================================= '''
services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 20))

# set each graphs' location
for i, item in enumerate(services):
    if i < 3: # 왼쪽 줄에 표시
        ax = data[item].value_counts().plot(kind='bar', ax=axes[i, 0], rot=0)

    elif i >= 3 and i < 6: # 가운데 줄에 표시
        ax = data[item].value_counts().plot(kind='bar', ax=axes[i - 3, 1], rot=0)

    elif i < 9: # 오른쪽 줄에 표시
        ax = data[item].value_counts().plot(kind='bar', ax=axes[i - 6, 2], rot=0)
    ax.set_title(item)

plt.show()

''' =========================================== Churn distribution graph ================================================= '''
ax = (target.value_counts()*100.0 /len(data)).plot(kind='bar', stacked = True, rot = 0, color=colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_xlabel('Churn')
ax.set_title('Churn Rate')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-3.5, \
            str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            weight = 'bold')

plt.show()

''' ================================================ Churn by various columns ==========================================='''

''' =========================================== Churn by gender ================================================= '''
gender_churn = df.groupby(['gender','Churn']).size().unstack()

ax = (gender_churn.T*100.0 / gender_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (8,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # tick (눈금 표시)

# set legend property ( location : center, title : churn, size : 14 ) (범례 설정)
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)

ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Gender',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart ( 막대 위에 숫자 표시 )
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()

''' =========================================== Churn by Ages ================================================= '''
seniorCitizen_churn = df.groupby(['SeniorCitizen','Churn']).size().unstack()

ax = (seniorCitizen_churn.T*100.0 / seniorCitizen_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Ages',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()
''' =========================================== Churn by Partner ================================================= '''
Partner_churn = df.groupby(['Partner','Churn']).size().unstack()

ax = (Partner_churn.T*100.0 / Partner_churn.T.sum()).T.plot(kind='bar',
                                                          width = 0.2,
                                                          stacked = True,
                                                          rot = 0,
                                                          figsize = (8,6),
                                                          color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Partner',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()

''' =========================================== Churn by Dependents ================================================= '''
Dependents_churn = df.groupby(['Dependents','Churn']).size().unstack()

ax = (Dependents_churn.T*100.0 / Dependents_churn.T.sum()).T.plot(kind='bar',
                                                            width = 0.2,
                                                            stacked = True,
                                                            rot = 0,
                                                            figsize = (8,6),
                                                            color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on dependents',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()

''' =========================================== Churn by Tenure ================================================= '''
ax = sns.kdeplot(df.tenure[(df["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(df.tenure[(df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Tenure')
ax.set_title('Distribution of Churn by tenure')

plt.show()
''' =========================================== Churn by PhoneService ================================================= '''
PhoneService_churn = df.groupby(['PhoneService','Churn']).size().unstack()

ax = (PhoneService_churn.T*100.0 / PhoneService_churn.T.sum()).T.plot(kind='bar',
                                                                  width = 0.2,
                                                                  stacked = True,
                                                                  rot = 0,
                                                                  figsize = (8,6),
                                                                  color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Phone Service Type',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by MultipleLines ================================================= '''
MultipleLines_churn = df.groupby(['MultipleLines','Churn']).size().unstack()

ax = (MultipleLines_churn.T*100.0 / MultipleLines_churn.T.sum()).T.plot(kind='bar',
                                                                      width = 0.2,
                                                                      stacked = True,
                                                                      rot = 0,
                                                                      figsize = (8,6),
                                                                      color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on MultipleLines',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by InternetService ================================================= '''
InternetService_churn = df.groupby(['InternetService','Churn']).size().unstack()

ax = (InternetService_churn.T*100.0 / InternetService_churn.T.sum()).T.plot(kind='bar',
                                                                        width = 0.2,
                                                                        stacked = True,
                                                                        rot = 0,
                                                                        figsize = (8,6),
                                                                        color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Internet Service',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by OnlineSecurity ================================================= '''
OnlineSecurity_churn = df.groupby(['OnlineSecurity','Churn']).size().unstack()

ax = (OnlineSecurity_churn.T*100.0 / OnlineSecurity_churn.T.sum()).T.plot(kind='bar',
                                                                            width = 0.2,
                                                                            stacked = True,
                                                                            rot = 0,
                                                                            figsize = (8,6),
                                                                            color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Online Security',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by OnlineBackup ================================================= '''
OnlineBackup_churn = df.groupby(['OnlineBackup','Churn']).size().unstack()

ax = (OnlineBackup_churn.T*100.0 / OnlineBackup_churn.T.sum()).T.plot(kind='bar',
                                                                          width = 0.2,
                                                                          stacked = True,
                                                                          rot = 0,
                                                                          figsize = (8,6),
                                                                          color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Online Backup',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by DeviceProtection ================================================= '''
DeviceProtection_churn = df.groupby(['DeviceProtection','Churn']).size().unstack()

ax = (DeviceProtection_churn.T*100.0 / DeviceProtection_churn.T.sum()).T.plot(kind='bar',
                                                                      width = 0.2,
                                                                      stacked = True,
                                                                      rot = 0,
                                                                      figsize = (8,6),
                                                                      color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Device Protection',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()


''' =========================================== Churn by TechSupport ================================================= '''
TechSupport_churn = df.groupby(['TechSupport','Churn']).size().unstack()

ax = (TechSupport_churn.T*100.0 / TechSupport_churn.T.sum()).T.plot(kind='bar',
                                                                              width = 0.2,
                                                                              stacked = True,
                                                                              rot = 0,
                                                                              figsize = (8,6),
                                                                              color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn',fontsize =14)
ax.set_ylabel('% Churn',size = 14)
ax.set_title('Distribution of Churn based on Technical Support',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                weight = 'bold',
                size = 14)

plt.show()
''' =========================================== Churn by StreamingTV ================================================= '''
tv_churn = df.groupby(['StreamingTV','Churn']).size().unstack()

ax = (tv_churn.T*100.0 / tv_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Streaming TV Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()

''' =========================================== Churn by StreamingMovies ================================================= '''
movies_churn = df.groupby(['StreamingMovies','Churn']).size().unstack()

ax = (movies_churn.T*100.0 / movies_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Streaming Movies Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()

''' =========================================== Churn by Contract ================================================= '''
contract_churn = df.groupby(['Contract','Churn']).size().unstack()

ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()
''' =========================================== Churn by PaperlessBilling (No : 종이 청구서, Yes : 전자청구서)  ================================================= '''
bill_churn = df.groupby(['PaperlessBilling','Churn']).size().unstack()

ax = (bill_churn.T*100.0 / bill_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Billing Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()

''' =========================================== Churn by PaymentMethod  ================================================= '''
payment_churn = df.groupby(['PaymentMethod','Churn']).size().unstack()

ax = (payment_churn.T*100.0 / payment_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Payment Method',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
               weight = 'bold',
               size = 14)

plt.show()

''' ===================================== Churn by Monthly Charges ================================= '''
ax = sns.kdeplot(df.MonthlyCharges[(df["SeniorCitizen"] == 'No')], color="Red", shade=True)
ax = sns.kdeplot(df.MonthlyCharges[(df["SeniorCitizen"] == 'Yes')], ax=ax, color="Blue", shade=True)

# set legend property ( set name of each value and location )
ax.legend(["Not Churn", "Churn"], loc='upper right')

ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')

plt.show()

''' =========================================== Churn by Total Charges ================================================= '''
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of total charges by churn')

plt.show()