# Airline Sentiment Analysis

## About the Project
This project analyzes Facebook comments from the pages of four major US airlines: [Delta](https://www.facebook.com/delta/), [Southwest](https://www.facebook.com/SouthwestAir/), [Spirit](https://www.facebook.com/SpiritAirlines/), and [American Airlines](https://www.facebook.com/AmericanAirlines/).  We look at these airlines' post frequency, the number of comments per post, and the sentiment of these comments. To analyze comment sentiment, we use an [existing roBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) finetuned on Twitter posts.  Interestingly, in this analysis, we can spot changes in user sentiment and airline posting behavior during disruptive events in the airline industry, like the [Southwest Airlines mass flight cancellations](https://en.wikipedia.org/wiki/2022_Southwest_Airlines_scheduling_crisis) of December 2022.

## Results

## Using this repository
First, install dependencies from the requirements.txt.  This can be done easily with pip:

`pip install -r requirements.txt`

## Project Structure
This project has a few notebooks and other python files used for data collection and analysis:
1) facebook-scraper.ipynb: scrapes facebook posts and comments from a given facebook page
2) sentiment-classifier.ipynb: analyzes the sentiment of comments on each post
3) data-analyzer.ipynb: breaks down collected data into charts.  For example number of posts per day, comments per post, number of positive/negative/neutral comments per post, etc.
4) roberta.py: contains a helper class for classifying sentiment with the roBERTa model via the huggingface transformers library

This repo also contains 2 data folders:
1) scraped_data: raw comments from airline facebook pages scraped with 
2) sentiment_data: processed data, containing the number of positive/negative/neutral comments on each post 

## Data Collection
To scrape from facebook, run the facebook-scraper.ipynb folder.  In the second cell, you can specify the name of a Facebook page, a number of posts, and a number of comments for scraping.  Also specify a Facebook account's username and password for scraping.  Please note that this account may be banned.

After the correct information is supplied, the notebook will scrape the desired posts, first without comments, and then with comments.  The data will be saved in the scraped_data folder.

## Sentiment Analysis
