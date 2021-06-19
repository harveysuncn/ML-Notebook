# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:30:05 2021

@author: Administrator
"""
import csv
import asyncio 
from pyppeteer import launch
from pyquery import PyQuery as pq
from urllib.parse import urljoin


def index_parser(current_url, content):
    doc = pq(content)
    urls = doc(".listed-game-block").items()
    game_urls = []
    next_url = ''
    for url in urls:
        game_urls.append(urljoin(current_url, url.attr("href")))
    return game_urls, next_url

def more_comment_parser(content):
    doc = pq(content)
    more_comment = doc(".fetch-more-action")
    return more_comment

    
def comment_parser(content):
    doc = pq(content)
    comments = []
    elements = doc(".player-post-main-content").children(".content")
    for element in elements:
        comments.append(element.text)
    
    return comments

def save_comment(game_name, comments):
    with open("comments.csv", 'a', newline='') as csvfile:
        fieldnames = ['game', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for comment in comments:
            try:
                writer.writerow({'game': game_name, 'comment': comment})
            except UnicodeEncodeError:
                continue
            

async def page_browser(url):
    pass


async def browser(url):
    brow = await launch({'headless': False})
    page = await brow.newPage()
    await page.goto(url)
    urls, next_url = index_parser(url, await page.content())
    
    for url in urls[1:]:
        await page.goto(url)
        await page.hover(".SK-footer-left")
        await page.waitFor(2*1000)
        more_comment = more_comment_parser(await page.content())
        while more_comment.text() == '查看更多评论':
            await page.click(".fetch-more-action")
            await page.waitFor(2*1000)
            more_comment = more_comment_parser(await page.content())
        
        #await page.evaluate('_ => {window.scrollBy(0, window.innerHeight);}')
        comments = comment_parser(await page.content())
        save_comment(await page.title(), comments)
        print(len(comments))
        
    #await brow.close()
    

index_url = "https://www.sonkwo.com/store/search"
asyncio.get_event_loop().run_until_complete(browser(index_url))