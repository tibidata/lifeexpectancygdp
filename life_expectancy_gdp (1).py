#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#THE GOAL OF THIS PROJECT IS TO PROVE THE HIGHER THE GDP THE HIGHER THE AVERAGE LIFE EXPECTANCY IS#


# In[1]:


#IMPORTING THE NECESSARY LIBRARIES

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# In[93]:


#GETTIN TO KNOW WITH THE DATA

df = pd.read_csv('all_data.csv')
print(df.head())

#MAKING THE COLUMN NAMES EASIER TO USE

df.rename(columns={'Country':'country','Year':'year', 'Life expectancy at birth (years)':'life_exp_years','GDP':'gdp'}, inplace=True)
print(df.head())

#CHECKING THE DATATYPES FOR FURTHER USE

print(df.dtypes)


df['gdp_int'] = df.gdp.apply(lambda x: int(x))
df['gdp_billion_dollar'] = df.gdp_int.apply(lambda x : x/1000000000)

del df['gdp_int']
del df['gdp']


#OUR DATA IS CLEAN AND READY TO USE


# In[222]:


#CHECKING THE DIFFERENCE IN THE GDP FOR THE YEARS IN THE 6 NATIONS
years=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]

chile = df[df['country'] == 'Chile']
china = df[df['country'] == 'China']
usa = df[df['country'] == 'United States of America']
zim = df[df['country'] == 'Zimbabwe']
mex = df[df['country'] == 'Mexico']
ger = df[df['country'] == 'Germany']



plt.figure(figsize=(10,6))
sns.set()
sns.set_style('white')
sns.set_palette('Pastel2')
sns.set_context('talk')
ax = plt.subplot()
ax.set_title('The GDPs compared between the countries')
plt.plot(years,chile['gdp_billion_dollar'],label='Chile')
plt.plot(years,china['gdp_billion_dollar'],label='China')
plt.plot(years,usa['gdp_billion_dollar'],label='United States of America')
plt.plot(years,zim['gdp_billion_dollar'],label='Zimbabwe')
plt.plot(years,mex['gdp_billion_dollar'],label='Mexico')
plt.plot(years,ger['gdp_billion_dollar'],label='Germany')
plt.legend()
plt.savefig('gdp_between_countries.png')
plt.show()


# In[223]:


#GDP FOR DIFFERENT NATIONS
sns.set()
sns.set_palette('Pastel1')
sns.set_style('white')
sns.set_context('talk')

plt.figure(figsize=(20,25))


ax1 = plt.subplot(3,2,1)
ax1.set_title('Chiles GDP over the last 15 years')
ax1.set_yticks([])
ax1 = plt.plot(years,chile['gdp_billion_dollar'],label='Chile')
plt.legend(loc='upper left')

ax2 = plt.subplot(3,2,2)
ax2.set_title('Chinas GDP over the last 15 years')
ax2.set_yticks([])
ax2 = plt.plot(years,china['gdp_billion_dollar'],label='China')
plt.legend()
               
ax3= plt.subplot(3,2,3)
ax3.set_title('The USA\'s GDP over the last 15 years')
ax3.set_yticks([])               
ax3 = plt.plot(years,usa['gdp_billion_dollar'],label='United States of America')
plt.legend()
               
ax4 = plt.subplot(3,2,4)
ax4.set_title('Zimbabwes GDP over the last 15 years')
ax4.set_yticks([])
ax4 = plt.plot(years,zim['gdp_billion_dollar'],label='Zimbabwe')
plt.legend()
               
ax5 = plt.subplot(3,2,5)
ax5.set_title('Mexicos GDP over the last 15 years')
ax5.set_yticks([])
ax5 = plt.plot(years,mex['gdp_billion_dollar'],label='Mexico')
plt.legend()
               
ax6 = plt.subplot(3,2,6)
ax6.set_title('Germanys GDP over the last 15 years')
ax6.set_yticks([])
ax6 = plt.plot(years,ger['gdp_billion_dollar'],label='Germany')
plt.legend()
plt.savefig('gdp_countries.png')
plt.show()


# In[224]:


#CHECKING THE AVERAGE LIFEEXPECTANCY OF THE NATIONS

countries_x = list(set(df['country']))
plt.figure(figsize=(8,10))
ax = plt.subplot()
ax.set_xticks(range(6))
ax.set_xticklabels(countries_x, rotation=270)
#ax.set_xlabel('Countries')
#ax.set_ylabel('Years')
ax.set_title('Average Life Expectancy In The Countries')
sns.set(style='whitegrid', palette='Pastel1', context='talk')
sns.barplot(data=df, x=df['country'], y='life_exp_years', ci=None)
plt.savefig('avg_life_exp.png')
plt.show()


# In[225]:


#COMPARING LIFE ECPECTANCY

sns.set(palette='Pastel1', style='whitegrid', context='talk')
plt.figure(figsize=(30,5))
sns.barplot(data=df, x='year', y='life_exp_years',hue='country')
plt.savefig('avg_life_exp_by_year.png')
plt.show()


# In[226]:


sns.set(palette="Pastel1", style='white', context='talk')
plt.figure(figsize=(10,8))
ax = plt.subplot()
ax.set_title('The life expectancy over the past 15 years')
ax.set_xlabel('Years')
ax.set_ylabel('Average life exepectancy')
#ax.title('The Life Expectancy Over The PAst 15 Years')
plt.plot(years, chile['life_exp_years'], label='Chile')
plt.plot(years, usa['life_exp_years'],label='USA')
plt.plot(years, zim['life_exp_years'], label='Zimbabwe')
plt.plot(years, china['life_exp_years'], label='China')
plt.plot(years, ger['life_exp_years'], label='Germany')
plt.plot(years, mex['life_exp_years'], label='Mexico')
plt.savefig('life_exp_lineplot.png')
plt.legend()
plt.show()


# In[ ]:





# In[227]:


#AVERAGE LIFE EXPECTANCY FOR THE LAST 15 YEARS



mean_life_exp = df.groupby('country').life_exp_years.mean().reset_index()
print(mean_life_exp)


sns.set(style='dark', palette='Pastel1', context="talk")
plt.figure(figsize=(10,8))
ax=plt.subplot()
ax.set_title('Average life expectancy for the past 15 years')
ax.set_xticklabels(list(set(df['country'])), rotation=60)
sns.barplot(data=df, x='country', y='life_exp_years')
plt.savefig('avg_life_exp_bar.png')
plt.show()


# In[228]:


#DISTRIBUTION FOR THE AVG LIFE EXPECTANCY AND THE AVERAGE GDP


sns.set(palette='Pastel1', style='dark', context='talk')
plt.figure(figsize=(10,8))
ax = plt.subplot()
sns.violinplot(data=df, x='country', y='life_exp_years')
plt.savefig('life_exp_dist_violin.png')
plt.show()


# In[230]:


#ZIMBABWE LOOKS INTERESTING SO WE SHOULD TAKE A CLOSER LOOK AT IT

sns.set(palette='Pastel1', style='dark', context='talk')
plt.figure(figsize=(10,8))
ax = plt.subplot()
plt.plot(zim['year'],zim['life_exp_years'],label='Life Expectancy')
plt.plot(zim['year'],zim['gdp_billion_dollar'], label='GDP (Billion Dollar)')
plt.legend()
plt.savefig('zim_gdp_vs_life_exp1.png')
plt.show()


sns.violinplot(data=zim,x='country',y='life_exp_years')
plt.savefig('zim_life_Exp_violin.png')
plt.show()
sns.boxplot(data=zim, x='country', y='life_exp_years')
plt.savefig('zim_life_exp_box.png')
plt.show()


# In[232]:


def gdp_vs_life_exp(df):
    
    n = 1  # This is our first dataset (out of 2)
    t = 2 # Number of dataset
    d = 16 # Number of sets of bars
    w = 0.8 # Width of each bar
    bars1_x = [t*element + w*n for element
             in range(d)]

    n = 2  # This is our first dataset (out of 2)
    t = 2 # Number of dataset
    d = 16 # Number of sets of bars
    w = 0.8 # Width of each bar
    bars2_x = [t*element + w*n for element
             in range(d)]
    
    country_name = df.iloc[0,0]
    #print(country_name)
    middle_x = [ (a + b) / 2.0 for a, b in zip(bars1_x, bars2_x)]
    file_name= df.iloc[0,0] + '_bar_plot.png'
    
    if 'Zimbabwe' in country_name: #ONLY FOR ZIMBABWE IT HAS SO SMALL GDP IMPOSSIBLE TO COMPARE W/O THIS MODIFICATION
        bar1_y = [x*5 for x in list(set(df['gdp_billion_dollar']))]
        
        ax = plt.subplot()
        ax.set_yticks([])
        ax.set_title(country_name)
        plt.xticks(middle_x, years)
        ax.set_xticklabels(years,rotation=270)
        plt.bar(bars1_x,bar1_y, label='GDP')
        plt.bar(bars2_x,df['life_exp_years'], label='Life Expectancy')
        plt.legend()
        plt.savefig(file_name)
        plt.show()
    else:
        bar1_y = [x/40 for x in list(set(df['gdp_billion_dollar']))]
        
        ax = plt.subplot()
        ax.set_yticks([])
        ax.set_title(country_name)
        plt.xticks(middle_x, years)
        ax.set_xticklabels(years,rotation=270)
        plt.bar(bars1_x,bar1_y, label='GDP')
        plt.bar(bars2_x,df['life_exp_years'], label='Life Expectancy')
        plt.legend()
        plt.savefig(file_name)
        plt.show()

gdp_vs_life_exp(zim)
gdp_vs_life_exp(usa)
gdp_vs_life_exp(ger)
gdp_vs_life_exp(china)
gdp_vs_life_exp(mex)
gdp_vs_life_exp(chile)

