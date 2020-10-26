#!/usr/bin/env python
# coding: utf-8

# # Recommender Systems with Python
# 

# ## Import Libraries

# In[136]:

import numpy as np
import pandas as pd


# ## Get the Data

# In[137]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[138]:


df.head()


# Now let's get the movie titles:

# In[139]:


movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()


# We can merge them together:

# In[140]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # EDA
# 
# Let's explore the data a bit and get a look at some of the best rated movies.
# 
# ## Visualization Imports

# In[160]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's create a ratings dataframe with average rating and number of ratings:

# In[142]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[143]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[144]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Now set the number of ratings column:

# In[159]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Now a few histograms:

# In[146]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[147]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[148]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# Okay! Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:

# ## Recommending Similar Movies

# Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[149]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[150]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

# In[161]:


ratings.head()


# Now let's grab the user ratings for those two movies:

# In[162]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# We can then use corrwith() method to get correlations between two pandas series:

# In[163]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# Let's clean this by removing NaN values and using a DataFrame instead of a series:

# In[164]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie). 

# In[155]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[165]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now sort the values and notice how the titles make a lot more sense:

# In[157]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# Now the same for the comedy Liar Liar:

# In[158]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# # Great Job!
