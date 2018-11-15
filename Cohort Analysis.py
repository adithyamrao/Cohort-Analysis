
# coding: utf-8

# # Cohort Analysis with Python
# ### What is cohort analysis?
# A cohort is a group of users who share something in common, be it their sign-up date, first purchase month, birth date, acquisition channel, etc. Cohort analysis is the method by which these groups are tracked over time, helping you spot trends, understand repeat behaviors (purchases, engagement, amount spent, etc.), and monitor your customer and revenue retention.
# 
# It’s common for cohorts to be created based on a customer’s first usage of the platform, where "usage" is dependent on your business’ key metrics. For Uber or Lyft, usage would be booking a trip through one of their apps. For GrubHub, it’s ordering some food. For AirBnB, it’s booking a stay.
# 
# With these companies, a purchase is at their core, be it taking a trip or ordering dinner — their revenues are tied to their users’ purchase behavior.
# 
# In others, a purchase is not central to the business model and the business is more interested in "engagement" with the platform. Facebook and Twitter are examples of this - are you visiting their sites every day? Are you performing some action on them - maybe a "like" on Facebook or a "favorite" on a tweet?1
# 
# When building a cohort analysis, it’s important to consider the relationship between the event or interaction you’re tracking and its relationship to your business model.
# 
# ### Why is it valuable?
# Cohort analysis can be helpful when it comes to understanding your business’ health and "stickiness" - the loyalty of your customers. Stickiness is critical since it’s far cheaper and easier to keep a current customer than to acquire a new one. For startups, it’s also a key indicator of product-market fit.
# 
# Additionally, your product evolves over time. New features are added and removed, the design changes, etc. Observing individual groups over time is a starting point to understanding how these changes affect user behavior.
# 
# It’s also a good way to visualize your user retention/churn as well as formulating a basic understanding of their lifetime value.
# 
# ** import modules **

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Customizing matplotlib

# In[3]:


plt.style.use('fivethirtyeight')

width, height = plt.figaspect(4)
fig = plt.figure(figsize=(width,height), dpi=400)
from matplotlib import rcParams
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'DejaVu Sans'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


# In[10]:


# Load csv
df = pd.read_csv('sessions_df.csv')
df.head(15)


# In[11]:


# count unique users
df['user_id'].nunique()


# **1. Create a period column based on the SessionDate** <br>
# 
# Since we're doing monthly cohorts, we'll be looking at the total monthly behavior of our users. Therefore, we don't want granular SessionDate data (right now).

# In[12]:


# cast 'session_date' as date
from datetime import datetime
df['session_date'] = pd.to_datetime(df['session_date'])


# In[13]:


# add a new column with order month 
# Lambda lets you define one-line mini-functions 
# time.strftime(format[, t]) 
# %m - month (01 to 12)

df['SessionPeriod'] = df.session_date.apply(lambda x: x.strftime('%Y-%m'))
df.head(10)


# #### 2. Determine the user's cohort group (based on their first session)
# 
# Create a new column called CohortGroup, which is the year and month in which the user's first visit occurred.

# In[14]:


# Set the DataFrame index (row labels) using one or more existing columns.
df.set_index('user_id', inplace=True)

# Create Cohort Group base on first session date of the user
df['CohortGroup'] = df.groupby(level=0)['session_date'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)


# In[15]:


user_tot = df.groupby(['SessionPeriod', 'user_id']).sum()
user_tot.reset_index(inplace=True)
user_tot.head()


# #### 3. Rollup data by CohortGroup & OrderPeriod
# 
# Since we're looking at monthly cohorts, we need to aggregate users, orders, and amount spent by the CohortGroup within the month (OrderPeriod).

# In[16]:


grouped = df.groupby(['CohortGroup', 'SessionPeriod'])
grouped_1=grouped
# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg({'user_id': pd.Series.nunique,
                       'session_id': pd.Series.nunique})

# make the column names more meaningful
cohorts.rename(columns={'user_id': 'TotalUsers',
                        'session_date': 'TotalSession'}, inplace=True)
cohorts.head()


# #### 4. Label the CohortPeriod for each CohortGroup
# 
# We want to look at how each cohort has behaved in the months following their first purchase, so we'll need to index each cohort to their first purchase month. For example, CohortPeriod = 1 will be the cohort's first month, CohortPeriod = 2 is their second, and so on.
# 
# This allows us to compare cohorts across various stages of their lifetime.

# In[17]:


def cohort_period(df):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime', inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)


# In[18]:


# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()


# In[19]:


user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention_abs = cohorts['TotalUsers'].unstack(0)
user_retention.head(10)
user_retention


# Finally, we can plot the cohorts over time in an effort to spot behavioral differences or similarities. Two common cohort charts are line graphs and heatmaps, both of which are shown below.
# 
# Notice that the first period of each cohort is 100% -- this is because our cohorts are based on each user's first purchase, meaning everyone in the cohort purchased in month 1.

# In[21]:


width, height = plt.figaspect(.6)
fig = plt.figure(figsize=(width,height), dpi=120)
#plt.text(-1.6, 19.0, s = 'Cohort Analysis (%)', fontsize = 14,  fontweight='bold')#", fontname='Ubuntu', fontsize=14, fontweight='semibold')
#plt.text(-1.6, 16.7, s = 'Cohort analysis is a subset of behavioral analytics that takes the data from a \ngiven dataset and rather than looking at all users as one unit, it breaks them \ninto related groups for analysis.', fontsize = 9,  fontweight='medium')
plt.title("Cohort Analysis (abs)", fontname='DejaVu Sans', fontsize=14, fontweight='bold')
sns.heatmap(user_retention_abs.T, mask=user_retention.T.isnull(), annot=True, fmt='g', cmap='coolwarm')


# In[22]:


width, height = plt.figaspect(.6)
fig = plt.figure(dpi=120)
#plt.text(-1.6, 19.0, s = 'Cohort Analysis (%)', fontsize = 14,  fontweight='bold')#", fontname='Ubuntu', fontsize=14, fontweight='semibold')
#plt.text(-1.6, 16.7, s = 'Cohort analysis is a subset of behavioral analytics that takes the data from a \ngiven dataset and rather than looking at all users as one unit, it breaks them \ninto related groups for analysis.', fontsize = 9,  fontweight='medium')

plt.title("Churn Analysis", fontname='DejaVu Sans', fontsize=14, fontweight='bold')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='0.00%', cmap='viridis')


# In[24]:


width, height = plt.figaspect(.6)
fig = plt.figure(dpi=150)


ax = user_retention[['2016-11','2016-12','2017-01', '2017-02','2017-03','2017-04','2017-05','2017-06']].plot(figsize=(11,6))
plt.title("Retention rate (%) per CohortGroup", fontname='DejaVu Sans', fontsize=20, fontweight='bold')

plt.xticks(np.arange(1, 16.1, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
ax.set_xlabel("CohortPeriod", fontsize=10)
ax.set_ylabel("Retention(%)", fontsize=10)
plt.show()


# In[25]:


ax = user_retention.T.mean().plot(figsize=(11,6), marker='s')
plt.title("Retention rate (%) per CohortGroup", fontname='DejaVu Sans', fontsize=20, fontweight='bold')

plt.xticks(np.arange(1, 16.1, 1), fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
ax.set_xlabel("CohortPeriod", fontsize=10)
ax.set_ylabel("Retention(%)", fontsize=10)
plt.show()

