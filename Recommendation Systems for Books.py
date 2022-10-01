#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all necessary library..
import numpy as np # for array calculations
import pandas as pd # for dataframe manipulations
import sqlite3 # sqlite3 database for storing data


# for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# for not write again and again to show the graph
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load dataframe
df = pd.read_csv('books.csv',error_bad_lines= False)
df.head()


# In[3]:


# creating a database and a table to store the data. 
conn = sqlite3.connect('Books_database')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS books (bookID, title, authors, average_rating, isbn, isbn13, language_code, num_pages, ratings_count, text_reviews_count,publication_date, publisher)')
conn.commit()


# In[4]:


# exporting the dataframe data to the SQLite database.
df.to_sql('books', conn, if_exists='replace', index = False)


# In[5]:


# Fetching data from Sqlite3 back to dataframe
conn = sqlite3.connect("Books_database")
dfback = pd.read_sql_query("SELECT * from books", conn)
print(dfback)


# # Descriptive Analysis

# In[6]:


# difffernt features..
df.columns


# ## Features Description:
# 1. bookID Contains the unique ID for each book/series
# 2. title contains the titles of the books
# 3. authors contains the author of the particular book
# 4. average_rating the average rating of the books, as decided by the users
# 5. ISBN ISBN(10) number, tells the information about a book - such as edition and publisher
# 6. ISBN 13 The new format for ISBN, implemented in 2007. 13 digits
# 7. language_code Tells the language for the books
# 8. Num_pages Contains the number of pages for the book
# 9. Ratings_count Contains the number of ratings given for the book
# 10. text_reviews_count Has the count of reviews left by users

# In[7]:


# check null values..
df.isnull().sum()


# There is no nan values in any of its attribute.

# In[8]:


# about dataframe 
df.info()


# In[9]:


#  numerical summary of dataframe 
df.describe()


# # Visualization

# ### Top 15 Rated Books

# In[10]:


top_fifteen = df[df['ratings_count'] > 1000000]
top_fifteen.sort_values(by='average_rating', ascending=False)
top_fifteen.head(15)


# As we can see above the top 15 rated books. We see that the maximum rating in our dataframe is 5.0 but we dont see any books in the above result with 5.0 rating. This is because we filtered these books on the basis of the number of ratings. We made sure that all the books that we have in the above results have a decent amount of rating. There can be books in the data that can have only 1 or 2 ratings can be rated 5.0. We want to avoid such books hence this sort of filtering.

# Let's go ahead and visualize this outcome in form of a graph.

# In[11]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 10))

data = top_fifteen.sort_values(by='average_rating', ascending=False).head(15)
gr = sns.barplot(x="average_rating", y="title", data=data, palette="CMRmap_r")

for i in gr.patches:
    gr.text(i.get_width() + .05, i.get_y() + 0.5, str(i.get_width()), fontsize = 10, color = 'k')
plt.show()


# ### Top 15 authors present in our data

# In[12]:


top_15_authors = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(15).set_index('authors')
top_15_authors.head(15)


# Let's go ahead and take a look at some top 15 authors present in our data. We will rank them according to the number of books they have written provided these books are present in the data.

# In[13]:


plt.figure(figsize=(15,10))
ax = sns.barplot(top_15_authors['title'], top_15_authors.index, palette='CMRmap_r')

ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
plt.show()


# According to our graphs, Stephen king and P.G. Wodehouse have the most number of books in the data. Both the authors have 40 books in our data set followed by Rumiko Takahashi and Orson scott Card.

# ## Relationship between average rating and rating count

# In[14]:


ax = sns.relplot(data=df,
                 x="ratings_count",
                 y="average_rating",
                 color = '#95a3c3',
                 sizes=(400, 600), 
                 height=7, 
                 marker='o')


# ## Language Distribution

# As we can see here most of the books have the language as english. So, in the features selection section we colud remove non english rows in the dataframe for accuracy.

# ### Top 15 publisher

# In[15]:


top_15_publisher = df.groupby('publisher')['title'].count().reset_index().sort_values('title', ascending=False).head(15).set_index('publisher')
top_15_publisher.head(15)


# To get more about the publisher using visualizations

# In[16]:


plt.figure(figsize=(15,10))
ax = sns.barplot(top_15_publisher['title'], top_15_publisher.index, palette='CMRmap_r')

ax.set_title("Top 15 publisher with most books")
ax.set_xlabel("Total number of books")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
plt.show()


# ## Distribution of average_rating

# In[17]:


df.average_rating = df.average_rating.astype(float)
fig, ax = plt.subplots(figsize=[15,10])
sns.distplot(df['average_rating'],ax=ax)
ax.set_title('Average rating distribution for all books',fontsize=20)
ax.set_xlabel('Average rating',fontsize=13)


# It almost follows gussian distributions curve. So, it is very good for model training.

# After comparing the average rating with the different columns, we can go ahead with using the language and the Rating counts for our recommender system. Rest other colummns weren't making much sense and using them might not help us in a big way so we can omit them

# # Feature Engineering

# In[18]:


df.columns


# ## 1. Imputation

# In[19]:


threshold = 0.7
#Dropping columns with missing value rate higher than threshold
df = df[df.columns[df.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
df = df.loc[df.isnull().mean(axis=1) < threshold]


# In[20]:


df.head()


# ## 2. Handling Outliers

# In[21]:


# correlation between the features
corrmat = df.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 


# Here we can see that text_reviews_count is highly correlated with ratings_count. So, we can use either of these features.

# In[22]:


df2 =df.copy()


# We will now create a new column called 'rating_between'. We will divide our average rating column into various categories such as rating between 0 and 1, 1 and 2 and so on. This will work as one of the features that we will feed to our model so that it can make better predictions.

# In[23]:


df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"


# In[24]:


df2.head()


# In[25]:


rating_df = pd.get_dummies(df2['rating_between'])
rating_df.head()


# In[26]:


l_code_df = pd.get_dummies(df2['language_code'])
l_code_df.head()


# In[27]:


## now we combine these two in the dataframe 

features = pd.concat([l_code_df, rating_df, df2['average_rating'], df2['ratings_count']], axis=1)
features.head()


# Now that we have our features ready, we will now use the Min-Max scaler to scale these values down. It will help in reducing the bias for some of the books that have too many features. It will basically find the median for all and equalize it,

# # Model Building

# In[28]:


# import necessary pakages for k-nearest-neighbour

from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[29]:


min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)


# In[30]:


model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)


# In[31]:


def book_recommendation_engine(book_name):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    book_id = book_id[0]
#     print('book_id', book_id)
    for newid in idlist[book_id]:
#         print(newid)
        book_list_name.append(df2.loc[newid].title)
#         print(new_data.loc[newid].title)
    return book_list_name


# # Examples

# ## Example 1

# Here we have a list of recommendations for the book 'Little Women'.

# In[32]:


book_list_name = book_recommendation_engine('Little Women')
book_list_name


# ## Example 2

# Here we have a list of recommendations for the book 'The Lord of the Rings: Complete Visual Companion'.

# In[33]:


book_list_name = book_recommendation_engine('The Lord of the Rings: Complete Visual Companion')
book_list_name

