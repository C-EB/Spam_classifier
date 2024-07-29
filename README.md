# Spam Classifier Project

## Objective

The objective of this project is to build a spam classifier using a dataset of ham (non-spam) and spam emails. The classifier will be trained to distinguish between ham and spam emails with high accuracy. 

## Dataset Description

The dataset used in this project is from the [SpamAssassin public corpus](http://spamassassin.apache.org/old/publiccorpus/). It consists of two sets of emails:

- **Ham emails:** Legitimate, non-spam emails.
- **Spam emails:** Unsolicited, often irrelevant or inappropriate emails.

The dataset contains 2500 ham emails and 500 spam emails.

## Project Workflow

1. **Fetching and Loading Data**
2. **Exploring the Data**
3. **Preprocessing Emails**
4. **Vectorizing Email Content**
5. **Training a Classifier**
6. **Evaluating the Classifier**

---

### Fetching and Loading Data

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url), ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]

ham_dir, spam_dir = fetch_spam_data()
```
This block of code downloads the dataset from the SpamAssassin public corpus, extracts it, and organizes it into directories.

**Output:**

```
Downloading datasets\spam\ham.tar.bz2
Downloading datasets\spam\spam.tar.bz2
```

### Loading Emails

```python
ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]

len(ham_filenames)
len(spam_filenames)
```

This code block loads the email filenames into lists and checks the number of emails in each category.

**Output:**

```
2500
500
```

### Parsing Emails

```python
import email
import email.policy

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(filepath) for filepath in ham_filenames]
spam_emails = [load_email(filepath) for filepath in spam_filenames]
```

This code block parses the emails using Python's `email` module.

### Example of Ham and Spam Email

```python
print(ham_emails[1].get_content().strip())
print(spam_emails[6].get_content().strip())
```

This code block prints an example of a ham email and a spam email to give a feel of what the data looks like.

**Output:**

```
Ham Email Content:
Martin A posted:
Tassos Papadopoulos, the Greek sculptor behind the plan, judged that the
limestone of Mount Kerdylio, 70 miles east of Salonika and not far from the
Mount Athos monastic community, was ideal for the patriotic sculpture. 
...

Spam Email Content:
Help wanted.  We are a 14 year old fortune 500 company, that is
growing at a tremendous rate.  We are looking for individuals who
want to work from home.
...
```

### Exploring Email Structures

```python
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email) for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()

from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

structures_counter(ham_emails).most_common()
structures_counter(spam_emails).most_common()
```

This code block explores the different structures of the emails.

**Output:**

```
Ham Email Structures:
[('text/plain', 2408), ...]
Spam Email Structures:
[('text/plain', 218), ('text/html', 183), ...]
```

### Email Headers

```python
for header, value in spam_emails[0].items():
    print(header, ":", value)

spam_emails[0]["Subject"]
```

This code block examines the headers of the emails and extracts the subject header.

**Output:**

```
Headers:
Return-Path : <12a1mailbot1@web.de>
...
Subject : Life Insurance - Why Pay More?
```

### Splitting Data

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This code block splits the data into a training set and a test set.

### Preprocessing Emails

```python
import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)
```

This code block converts HTML content to plain text.

**Example Output:**

```
<HTML><HEAD>...</HTML> becomes:
OTC
Newsletter
Discover Tomorrow's Winners
...
```

### Email to Text Conversion

```python
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

print(email_to_text(sample_html_spam)[:100], "...")
```

This function converts an email to plain text regardless of its format.

**Example Output:**

```
OTC
Newsletter
Discover Tomorrow's Winners
...
```

### Stemming

```python
import nltk

stemmer = nltk.PorterStemmer()

for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
    print(word, "=>", stemmer.stem(word))
```

This code block demonstrates stemming using the NLTK library.

**Output:**

```
Computations => comput
Computation => comput
Computing => comput
...
```

### Email to Word Counter Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True, replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in

 word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

from urlextract import URLExtract

url_extractor = URLExtract()
```

This class transforms emails into word count vectors.

**Example Output:**

```python
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts
```

**Output:**

```
array([Counter({...}), Counter({...}), Counter({...})], dtype=object)
```

### Word Counter to Vector Transformer

```python
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors.toarray()
```

This class transforms word counts into sparse vectors.

**Output:**

```
array([[ 6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [99, 11,  9,  8,  3,  1,  3,  1,  3,  2,  3],
       [67,  0,  1,  2,  3,  4,  1,  2,  0,  1,  0]], dtype=int32)
```

### Training the Classifier

```python
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
score.mean()
```

This code block trains a Logistic Regression classifier using a preprocessing pipeline.

**Output:**

```
0.985
```

### Evaluating the Classifier

```python
from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
```

This code block evaluates the classifier on the test set.

**Output:**

```
Precision: 96.88%
Recall: 97.89%
```

---

This detailed `README.md` should help provide clear documentation and explanation for each step of your spam classifier project.
