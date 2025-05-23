# Exercise
# Using the realnewslike subset of C4, prepare a word frequency counter, counting the number of times each word appears in the dataset. To make it simple, define a word as a contiguous sequence of characters separated by white space. Remove frequent function words (called stop words in NLP) like “the,” “is,” etc. from your analysis. What topics seem to be underrepresented or overrepresented?

from datasets import load_dataset
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

realnewslike = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")

article_text = ""
for i, example in enumerate(realnewslike):
    if "India" in example["text"]:
        article_text += example["text"]
    if i > 100:
        break

article_text_in_lower = article_text.lower()
words = article_text_in_lower.split()
word_counts = {}

for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
        

filtered_word_counts = {}

for word, count in word_counts.items():
    if word not in stop_words:
        filtered_word_counts[word] = count
        
sorted_filtered_counts = sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)
print(sorted_filtered_counts)