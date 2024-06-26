#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:38:21 2024

News Sentiment Analysis

@author: Muykheng Long
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


url = 'https://www.investing.com/news/stock-market-news'
request = requests.get(url)
soup = BeautifulSoup(request.text, 'html.parser')


url_list = []
headlines = []
date_time = []
news_text = []

for i in range(1,3):
    if i == 1:
        url = 'https://www.investing.com/news/stock-market-news'
    else: 
        url = f'https://www.investing.com/news/stock-market-news/{i}'
    
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'html.parser')
    articles = soup.find_all('div',{'class':'news-analysis-v2_content__z0iLP w-full text-xs sm:flex-1'})
    for article in articles:
        a_tag = article.find('a',{'class':'text-inv-blue-500 hover:text-inv-blue-500 hover:underline focus:text-inv-blue-500 focus:underline whitespace-normal text-sm font-bold leading-5 !text-[#181C21] sm:text-base sm:leading-6 lg:text-lg lg:leading-7'})
        if a_tag:
            url_list.append(a_tag['href'])
            headlines.append(a_tag.text)


for www in url_list: 
    
    url = f'https://www.investing.com{www}'
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'html.parser')
    
    # extract date published 
    published_datetime = soup.find('div',{'class':'flex flex-col gap-2 text-warren-gray-700 md:flex-row md:items-center md:gap-0'}).text[11:]
    date_time.append(published_datetime)
        
    # extract news content 
    temp = []
    
    t_paragraphs = soup.find_all('p')
    for paragraph in t_paragraphs:
        temp.append(paragraph.text)
        
    for last_paragraph in reversed(temp):
        if last_paragraph == '***':
            break
 
    if temp.index(last_paragraph) != 0: 
        joined_text = ''.join(temp[:temp.index(last_paragraph)])
    else:
        joined_text = ''.join(temp)

    news_text.append(joined_text)

# convert to dataframe
news_df = pd.DataFrame({'Date': date_time,
                        'Headline': headlines,
                        'News': news_text,
                        })

news_df.to_csv('news_df.csv',)
# use VADER to perform sentiment analysis
analyser = SentimentIntensityAnalyzer()

def comp_score(text):
    return analyser.polarity_scores(text)['compound']

news_df['Sentiment'] = news_df['News'].apply(comp_score)

news_df.to_csv('train_data')