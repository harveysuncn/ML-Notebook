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
from pyppeteer.errors import TimeoutError


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

def no_comment_parser(content):
    doc = pq(content)
    no_comment = doc(".to-bottom")
    return no_comment
    
def comment_parser(content):
    doc = pq(content)
    comments = []
    elements = doc(".player-post-main-content").children(".content")
    for element in elements:
        comments.append(element.text)
    
    return comments

def save_comment(game_name, comments):
    with open("sonkwo_comments.csv", 'a', newline='') as csvfile:
        fieldnames = ['game', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for comment in comments:
            try:
                writer.writerow({'game': game_name, 'comment': comment})
            except UnicodeEncodeError:
                continue
            

async def page_browser(url, browser):
    try:
        page = await browser.newPage()
        await page.goto(url)
        await page.waitForSelector('.SK-footer-left');
        await page.hover(".SK-footer-left")
        await page.waitFor(2*1000)
        
        more_comment = more_comment_parser(await page.content())
        while more_comment.text() == '查看更多评论':
            await page.click(".fetch-more-action")
            await page.waitFor(2*1000)
            more_comment = more_comment_parser(await page.content())
            no_comment = no_comment_parser(await page.content())
            if no_comment:
                break
            
        
        #await page.evaluate('_ => {window.scrollBy(0, window.innerHeight);}')
        comments = comment_parser(await page.content())
        title = await page.title()
        save_comment(title, comments)
        print(title, len(comments))
    except TimeoutError:
        print("url time exceed", url)
    finally:
        await page.close()


async def browser(url):
    brow = await launch({'headless': True})
    page = await brow.newPage()
    await page.goto(url)
    urls, next_url = index_parser(url, await page.content())
    tasks = []
    
    for url in urls:
        tasks.append(page_browser(url, brow))
    await asyncio.gather(*tasks)
        
    await brow.close()

if __name__ == '__main__':    
    # https://www.sonkwo.com/store/search?order=desc&page=30&sort=wishes_count
    
    for i in range(1, 2):
        index_url = "https://www.sonkwo.com/store/search?order=desc&page={}&sort=wishes_count".format(i)
        print("Page:", i)
        asyncio.get_event_loop().run_until_complete(browser(index_url))