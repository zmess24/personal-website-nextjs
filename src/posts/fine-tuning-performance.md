---
title: "How to measure performance of a fine-tuned LLM"
description: "Prompt engineering vs fine-tuning..."
tags: ["Coding"]
date: "12-27-2023"
link: "/posts/fine-tuning-performance"
---

I have been playing around a lot recently with different LLM models, and to test my knowledge, I decided to build a proof-of-concept application that leverages an LLM in a fun, focused, but intuitive way using OpenAI's ChatGPT API. As a JuJitsu practitioner who constantly records techniques I watch on YouTube into the notes section of my phone, the objective of my project was simple - given a YouTube video link, use ChatGPT to:

-   Perform sentiment analysis on whether or not the provided YouTube link is a JuJitsu instructional video.
-   Download the video's audio output into a text transcript.
-   Summarize the text transcript into a step-by-step breakdown of the technique(s) covered in the video.

The finished application, which I called <a href="https://www.grapplegenius.com" target="_blank" ref="noreferrer">GrappleGenius</a>, was fun to build, but I learned some valuable lessons along the way, specifically around the topic of prompt engineering vs fine-tuning. The goal of this post is to dive into the difference between these two forms of output modification, as well as provide some techniques for evaluating the performance of a fine-tuned model vs a prompt engineered one relative to a specific task & dataset. Let's get started!

### But First, What is Prompt Engineering?

Prompt engineering is a process in which specific and carefully structured prompts are crafted to effectively communicate with a LLM (like ChatGPT, Claude, Gemini, etc) in a way that guides the model towards producing a desired output. The goal of prompt engineering is to maximize the accuracy and relevance of the LLM's responses without changing the actual weights of the underlying neural network itself.

For GrappleGenius, I leveraged prompt engineering to perform sentiment analysis on the video titles I provided to determine if they were related to JuJitsu or not, and also for summarizing the output of the video transcripts into a specific JSON format that could be consumed by my front-end.

Below is the system role I prompt engineered to perform the sentiment analysis:

```python
system_role = "You are a helpful sentiment analysis assistant whose sole purpose is to determine if the provided YouTube video titles are Brazilian Ju-Jitsu, Judo, or Wrestling instructional videos. I only want you to give 'True' or 'False' answers with no additional information."
```

As a rule of thumb, optimizing the output of an LLM for a specific task through prompt engineering is generally considered a recommended first approach before resorting to fine-tuning, due to the technical overhead involved in the later.

### What is Fine-Tuning?

Fine-tuning refers to the process of using additional data to further train an LLM by 'tuning' the weights of the underlying neural network to have a more nuanced understanding of the provided dataset, which can improve performance by producing more relevant results. Fine-tuning is often useful for business specific tasks that require domain experience, such as text classification and customer support chatbots.

For GrappleGenius, fine-tuning became a necessary step if I wanted ChatGPT to accurately classify YoutTube videos as JuJitsu intructionals or not, because the base model didn't natively have a deep enough of understanding of the language and syntax of YouTube vidoes titles to provide confident answers.

For the purposes of this blog post, we are going to discuss how to fine-tune OpenAI's `gpt-3.5-turbo` model.

### 1. Fine-Tuning Setup

First, we need to perform some initial setup by importing a few libraries. We'll start with the most important, the <a href="https://platform.openai.com/docs/overview" target="_blank" ref="noreferrer">OpenAI API client</a>, which we will use to perform the sentiment analysis. Then we'll import the official <a href="https://developers.google.com/youtube/v3" target="_blank" ref="noreferrer">YouTube API</a>, which we will use to create our dataset.

Please note that if you are looking to follow along, you will need to set up developer accounts with both platforms and download your API keys into a `.env` file so that they can be safely imported. You should never explicitly write out your API keys in plain text in a location that is publicly accessible.

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

Next, we are going to write a few helper methods for the purposes of creating our dataset and performing sentiment analysis on it. You can see my explanations of what each function is designed to do in the code comments:

```python
def yt_get_titles(query):
    '''
    1. Perform a search on YouTube for provided 'query' and return top 10 results.
    2. For each of the 10 titles returned, sanitize the outputs.
    3. Return all 10 santized titles in a list.
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
    1. Accepts a queue of search terms, and iterates over each term 1 by 1.
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
    '''
    1. For each user prompt in the 'messages' list, return GTP's response.
    2. If 'validation' is off, create 'assistant' dictionary with GPT's response as the 'content', and append to the original message. After iterating through each message, return original 'messages' list.
    3. If 'validation' is on, add '1' or '0' to the predictions list depending on if response is 'True' or 'False' for use 'print_accuracy_reports' function.
    '''
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
    '''
    1. Create sklearn's confusion matrix to evalute the accuracy of the classification task of the predictions against test labels.
    2. Create sklearn's classification report to measure the precision, recall, and f1 scores of the predictions against test lables.
    '''
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

It's always a good idea to diversify the samples in your dataset in order to make the training or tuning of a model more powerful and dynamic. For GrappleGenius, that meant I needed a mix of video titles that I wanted it to classify as `True` or `False`. I've added comments for which search terms I expected to fall into each category. The output of our `yt_query_search_terms` function will return a result of 150 search video titles.

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

Next is arguably the most tedious part of this process - labeling our data. There are many curated datasets that can be found online with pre-labeled outputs that don't need to be validated, but given we are creating a dataset from scratch, this is a step we will need perform ourselves. In lieu of labeling all 150 examples we've collected by hand, we can use ChatGPT to do a quick first pass and then simply validate the results of its output.

To do this, we first give ChatGPT a system role or "prompt" for how we want it to behave, and then we provide it with the content we want it to evaluate. For more information on the API input format ChatGPT expects, check out their <a href="https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset" target="_blank" ref="noreferrer" >official documentation</a>.

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

Next, we will divide our dataset into training & test sets using an 80/20 split, and save them to local files. The training data is what we will use to perform the fine-tuning, while the test data is what we will use to measure the improvement of performance relative to the baseline. Given we have 150 total instances in our dataset, that means our training set will include 120 examples, while our test set will include 30 examples.

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

After the data has been downloaded into separate files (e.g `data/test.json` & `data/train.json`), we can simply lookover the results of the initial sentiment analysis and overwrite the output where needed. And with that, we're ready to fine-tune!

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

Finally, let's validate the performance of our newly created fine-tuned model relative to a model that only uses prompt engineering. We will accomplish this by evaluating both models on the test data we previously created after some initial data preparation.

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

Amazing! Using only 120 training examples, the fine-tuned model is observing an almost ~13% increase in performance compared to the prompt engineered model! With an even larger and more diverse set of training data, it's probable to assume that we could boost performance even higher.

### Conclusion

In conclusion, we've seen how fine-tuning a LLM can lead to some significant increases in performance when the situation calls for it, and while this was a simple example that was easy to validate, the framework I used can be scaled up to a variety of use cases. I'm excited to build on this experience by working on a more complex task that is less straight forward to measure, such as peforming RAG (retrieval augmented generation) on a collection of provided documents.

I hope you enjoyed reading this post. I'll catch you next time!
