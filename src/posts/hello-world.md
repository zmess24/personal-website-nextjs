---
title: "Hello World! And welcome to my blog"
description: "My first post!"
tags: ["Business"]
date: "12/20/2023"
link: "/posts/hello-world"
image: "/images/GrappleGenius.png"
---

### This is an Intro Section:

Next.js has two forms of pre-rendering: **Static Generation** and **Server-side Rendering**. The difference is in **when** it generates the HTML for a page.

-   **Static Generation** is the pre-rendering method that generates the HTML at **build time**. The pre-rendered HTML is then _reused_ on each request.
-   **Server-side Rendering** is the pre-rendering method that generates the HTML on **each request**.

Importantly, Next.js lets you **choose** which pre-rendering form to use for each page. You can create a "hybrid" Next.js app by using Static Generation for most pages and using Server-side Rendering for others.

```python
import os
print("hello World!")
```

```python
def yt_get_titles(query):
    '''
    This purposed of this function is to retrieve the top 10 results from the YouTube Search API
    for every query given as an argument.
    '''
    titles = []

    print(f"Getting Seach Results for {query}")
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(part='snippet', type='video', q=query, maxResults=10)
    response = request.execute()

    for video in response["items"]:
        title = video["snippet"]["title"].replace("&#39;", "'").replace("&amp;", '&')
        titles.append({ "prompt": title, "completion": "" })

    return titles
```

```javascript
function test() {
	console.log("What's Going on dawg");
}
```
