---
title: "Measuring improvment of a fine-tuned LLM"
description: "Prompt engineering vs fine-tuning..."
tags: ["Coding"]
date: "12/27/2023"
link: "/posts/fine-tuning-performance"
---

Like many developers now-a-days, I have been playing around a lot recently with the LLM ecosystem and decided to build an experimental application dedicated to leveraging these powerful technologies in a fun but intuitive way. As a JuJitsu pracitionar, the goal of my application was simple - given a YouTube video link, use an LLM to:

-   Perform sentiment analysis on whether or not the provided link (e.g "https://www.youtube.com/watch?v=b8b9tR21K8Q") was related to JuJitsu.
-   Download & transcribe the video's audio output to text.
-   Summarize the video transcript by providing a summary and step-by-step breakdown of the techniques covered in the video that user's of my application could refer back to.

While I succeeded in building this application (which I called GrappleGenius), I learned some valuable lessons along the way, specifically aroud the topic of prompt engineering vs fine-tuning. The goal of this post is to dive into the difference between these two forms of output modification, as well as provide some methods for evaluating the performance of a fine-tuned model vs a prompt engineered one relative to a specific task & dataset. Let's get started!

### What is Prompt Engineering?

Prompt engineering is a process in which specific and carefully structured prompts are crafted to effectively communicate with an LLM (like ChatGPT, Claude, Gemini, ect) in a way that guides the model towards producing a desired output. The goal of prompt engineering is to maximize the accuracy and relevance of the LLM's responses without changing the actual weights of the neural network itself.

In GrappleGenius, I used prompt engineering to perform sentiment analysis on the video titles I provided to determine if they were related to JuJitsu or not, and also for summarizing the output of the video transcripts into consistently formatted JSON so they they could be consumed by my front end.

For example, below is the system prompt I used for performing sentiment analysis in GrappleGenius:

```python
system_role = "You are helpful sentiment analysis assistant whose sole purpose is to determine if the provided YouTube video titles are Brazilian Ju-Jitsu, Judo, or Wrestling instructionial videos. I only want you to give 'True' or 'False' answers with no additional information."
```

As a rule thumb, optimizing the output of an LLM for a specific task through prompt engineering is generally considered a recommended first approach before resorting to fine-tuning due to the technical overhead invovled in the later.

### What is Fine-Tuning?

Fine-tuning refers to the process of using additional data to further train a pretrained LLM by 'tuning' the weights of the neural network to have a more nuanced understanding of the provided dataset, which can imporve performance by producing faster and more relevant results. Fine-tuning is often useful for business specific tasks that require domain experience, such as text classification, interactive customer support chatbots, and sentiment analysis.

For GrappleGenius, fine-tuning became a neccesary step if I wanted my application to correctly classify Jujitsu videos, because ChatGPT didn't natively have a deep enough of understanding of YouTube video titles that would constitute a `True` or `False` class, leading to inconsistent results. I observed many instances during my during QA process where the same video titles would sometimes pass validation, while other times not.

For the purposes of this blog post, we are going to discuss how to fine-tune OpenAI's `gpt-3.5-turbo` model. Let's get into it!

### 1. Fine-Tuning Setup

First, we need to perform some initial setup by importing a few libraries, most importantly the <a href="https://platform.openai.com/docs/overview" target="_blank" ref="noreferrer">OpenAI API client</a> (which we will use to perform the sentiment analysis), and the official <a href="https://developers.google.com/youtube/v3" target="_blank" ref="noreferrer">YouTube API</a> (which we will use to create our dataset).

Please note that if you are looking to follow along, you will need to setup developer accounts with both platforms and download your API keys into a `.env` file so that they can be safely imported. You should never explicitly write out your API keys in plain text in a location that is publicly accessible.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.metrics import classification_report, confusion_matrix
load_dotenv()

# Initialize YouTube API Client
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize OpenAI API Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
```

### 2. Helper Methods

Next, we are going to write a few helper methods for the purposes of creating our dataset and performing the actual sentiment analysis on that data. You can see my explanations of what each function is designed to do in the code comments:

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

def yt_query_search_terms(search_terms, all_titles=[]):
    '''
    This functions purpose is to:
        1. Iterate over every search term in the provided search term list.
        2. For each term, grab the top 10 results from the 'get_titles' function.
        3. Append them to the output 'all_titles' array and return once every term has been queried.
    '''
    current_query = search_terms.pop()
    titles = get_titles(current_query)
    all_titles = all_titles + titles
    search_terms_left = len(search_terms)

    if search_terms_left != 0:
        return get_all_titles(search_terms, all_titles)
    else:
        return all_titles

def openai_perform_request(messages, model="gpt-3.5-turbo", validation=False):
    print("Getting responses. This may take a few minutes...")
    predictions = []

    for message in messages:
        # Perform API Call
        completion = client.chat.completions.create(
              model=model,
              messages=message["messages"]
        )

        response = completion.choices[0].message.content

        if validation == False:
            assistant = { "role": "assistant", "content": response }
            message["messages"].append(assistant)
        else:
            y = 1 if response == "True" else 0
            predictions.append(y)

    print("Finished")
    return messages if validation == False else predictions

def print_accuracy_reports(predictions, labels):
    print("Confusion Matrix:")
    print("")
    print(confusion_matrix(labels, predictions))
    print("")
    print("Classification Report:")
    print("")
    print(classification_report(labels, predictions))
```

### 3. Collect & Create Dataset of Titles from YouTube

Now, let's create our dataset set using the helper methods we wrote above.

It's always a good idea to diversify the samples in your dataset in order to make the tuning of the model more powerful. For GrappleGenius, that meant I needed a collection of both video titles that I wanted it to classify as `True`, and video titles that I wanted it to classify as `False`. I've added comments for which search terms I roughly expected to fall into each category. The output of our `yt_query_search_terms` function should return a result of 150 search video titles.

```python
# Initialize list of search terms and array of video titles.
search_terms = [
    "kimura trap attacks", # True
    "ashi garami entries", # True
    "bjj triangle escape", # True
    "bjj americana", # True
    "john danaher ankle lock", # True
    "guillotine from turtle", # True
    "darce choke", # True
    "double leg takedown", # True
    "b team", # False
    "gordon ryan vs dillon danis", # False
    "uchi mata judo", # True
    "blast double leg takedown", # True
    "arm drag to single leg takedown", # True
    "ankle pick takedown", # True
    "cross collar takedown" # True
]

video_titles = yt_query_search_terms(search_terms)
```

### 4. Label, Preprocess and Split Data into Test & Training Sets

Next is arguably the most tedious part of this process - labeling our data. There are many curataed datasets that can be found online with pre-labled outputs that doesn't need to be validated, but given we are creating a dataset from scratch, this is a step we will need peform ourselves. To help with this task and instead of labeling all 150 examples we've collected by hand, we can use ChatGPT to do a quick first pass and then just valdiate the results of it's output.

To do this, we first give ChatGPT a system role or "prompt" for how we want it to behave, and then we provide it with the content we want it to evaluate. For more information on the input format ChatGPT expects to it's API, check out their <a href="https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset" target="_blank" ref="noreferrer" >official documentation</a>.

```python
data = [];

# System prompt
system_role = "You are helpful sentiment analysis assistant whose sole purpose is to determine if the provided YouTube video titles are Brazilian Ju-Jitsu, Judo, or Wrestling instructionial videos. I only want you to give 'True' or 'False' answers with no additional information."

# Preprocess our video titles into a format ChatGPT can intake:
# e.g {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
for video in video_titles:
    system = {"role": "system", "content": system_role }
    user = {"role": "user", "content": video }
    data.append({"messages": [system, user]})

# Use ChatGPT to generate initial label outputs
data = openai_perform_request(data)
```

Next up, we will split our dataset into training & test sets using an 80/20 split and save them to local files. The training data is what we will use to perform the fine-tuning, while the test data is what we will use to measure the improvement of performance relative to the baseline. Given we have 150 total instances in our dataset, that means our training set will include 120 examples, while our test set will include 30 examples.

```python
import random
import math
import json

# Shuffle data
random.shuffle(data)

# Establish Training & Test Sizes
train_size = math.ceil(len(data) * 0.80)
test_size = math.ceil(len(data) * 0.20)

# Split data
train_data = data[:train_size]
test_data = data[(len(data) - test_size):]

# Create train & test jsonl files.
with open("data/test.jsonl", "w") as file:
    for line in test_data:
        file.write(json.dumps(line))
        file.write("\n")
    file.close()

with open("data/train.jsonl", "w") as file:
    for line in train_data:
        file.write(json.dumps(line))
        file.write("\n")
```

After the data has been downloaded into seperate files (e.g `data/test.json` & `data/train.json`), we can simply lookover the results of the initial sentiment analysis and overwrite the output where needed. And with that, we're ready to fine-tune!

### 5. Fine-Tune ChatGPT Model

Now the fun begins! Let's upload our training data to OpenAI so that it can be used to fine-tune our model and then initialize the fine-tuning process. Note that this may take a while to complete depending on the size of the training set.

```python
# First, we need to upload our training data to OpenAI.
# This will return a training file ID: 'file-[training_file_id]'
client.files.create(
  file=open("data/train.jsonl", "rb"),
  purpose="fine-tune"
)

# Second, we can start to fine tune a ChatGPT model.
# This will return a fine-tuned ID: 'ft:gpt-3.5-turbo-[ID]:personal:grapple-genius:[fine_tuned_id]'
client.fine_tuning.jobs.create(
    training_file="file-[training_file_id]",
    model="gpt-3.5-turbo",
    suffix="grapple-genius"

)
```

If we want to check on the status of training job, we can do so by passing in our training job ID (which is returned from the `create` method) like so:

```python
# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve("ftjob-[training_job_id]")
```

### 6. Validate & Measure Performance of Fine-Tuned Model

Finally, let's validate the performance of our newly created fine-tuned model relative to a model that only uses prompt engineering - the same model we used to initially create our test labels! We will accomplish this by evaluating both models on the test data we previously created after some initial data preperation.

```python
# Import our test data from our previously created file
with open("data/test.jsonl") as f:
    test_data = [json.loads(line) for line in f]
    f.close()

# Create input (x_test) & output (y_test) lists
y_test = []
x_test = []

# Iterate over each example in our test data, and seperate into input (e.g x_test) and output (e.g y_test) lists
for message in test_data:
    y = 1 if message["messages"][2]["content"] == "True" else 0
    system = message["messages"][0]
    user = message["messages"][1]
    y_test.append(y)
    x_test.append({ "messages": [system, user]})

# Evaluate fine-tuned GPT predictions against prompt GPT predictions using our input data
ft_predictions = openai_perform_request(x_test, "ft:gpt-3.5-turbo-[id]:personal:grapple-genius:[fine_tuned_id]", validation=True)
prompt_predictions = openai_perform_request(x_test, validation=True)

# Evalute performance
print("Fine-Tuned Metrics")
print("--------")
print_accuracy_reports(ft_predictions, y_test)
print("Prompt Metrics")
print("--------")
print_accuracy_reports(prompt_predictions, y_test)
```

This will return the below output (your own results may vary slightly):

```
Fine-Tuned Metrics
--------
Confusion Matrix:

[[ 5  2]
 [ 1 22]]

Classification Report:

              precision    recall  f1-score   support

           0       0.83      0.71      0.77         7
           1       0.92      0.96      0.94        23

    accuracy                           0.90        30
   macro avg       0.88      0.84      0.85        30
weighted avg       0.90      0.90      0.90        30

Prompt Metrics
--------
Confusion Matrix:

[[ 5  2]
 [ 8 15]]

Classification Report:

              precision    recall  f1-score   support

           0       0.38      0.71      0.50         7
           1       0.88      0.65      0.75        23

    accuracy                           0.67        30
   macro avg       0.63      0.68      0.62        30
weighted avg       0.77      0.67      0.69        30
```

Amazing! Using only 120 training examples, the **fine-tuned model (~90% accuracy)** is observing an almost **~13% increase in performance** compared to the **prompt engineered model (~77% accuaracy)** on the same test set of data! With an even a larger and more diverse set of training data, it's probable to assume that we could boost performance even higher.

### Conclusion
