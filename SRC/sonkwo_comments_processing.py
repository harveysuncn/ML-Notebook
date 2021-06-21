# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:19:04 2021

@author: Administrator
"""

from pandas import read_csv
from snownlp import SnowNLP



class Comments:
    def __init__(self):
        filepath = "C:\\Users\\Administrator\\ML-Notebook\\Datasets\\sonkwo_comments_utf8.csv"
        self.raw = read_csv(filepath, header=None)
    
    def games_comments(self):
        return self.raw
    
    def unique_games(self):
        game_list = set(self.raw[0])
        return game_list
        
    
def calc_sentiment(sentence):
    try:
        s = SnowNLP(sentence)
    except TypeError:
        return 0
    return s.sentiments


def games_sentiment():
    comments = Comments()
    game_list = comments.unique_games()
    game_comment_pair = comments.games_comments()
    
    comments_amount = dict()
    game_senti = dict()
    for game in game_list:
        # init game:sentiment dict
        game_senti[game] = 0
        comments_amount[game] = 0
    
    games = game_comment_pair[0]
    comments = game_comment_pair[1]
    for game_name, comment in zip(games, comments):
        
        senti_score = calc_sentiment(comment)
        game_senti[game_name] += senti_score
        if senti_score > 0:
            comments_amount[game_name] += 1
    
    avg = dict()
    for game in game_list:
        avg[game] = game_senti[game]/(comments_amount[game]+1e-15) # division by zero
    return sorted(avg.items(), key=lambda kv:(kv[1], kv[0]))
    

if __name__ == '__main__':
    res = games_sentiment()
    print(res)