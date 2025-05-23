# EXERCISE

sample_text = "Breaking news today: a fluffy cat and a playful dog are best friends! The cat, named Whiskers, loves to chase the dog. The dog, named Buddy, enjoys the game. This is happy news. In other news, the local government discussed city planning and new park rules. But the main story is the cat and the dog."

# step 1: convert the text to lowercase and split it into words
sample_text_in_lower = sample_text.lower()
words = sample_text_in_lower.split()
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

print("Word counts:")
print(word_counts)

# OR
# we can use the Counter class from the collections module to count the frequency of each word in python 
# from collections import Counter
# word_counts = Counter(words)
# print(word_counts)

# step 2: remove stop words
stop_words = ["a", "the", "is", "to", "and", "in", "on", "this", "was", "but", "of", "for", "with", "as", "by", "on", "at", "from", "up", "about", "into", "over", "after", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "some", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

filtered_word_counts = {}

for word, count in word_counts.items():
    if word not in stop_words:
        filtered_word_counts[word] = count
        
print("Filtered word counts:")
print(filtered_word_counts)

# OR
# we can use the Counter class from the collections module to count the frequency of each word in python 
# from collections import Counter
# word_counts = Counter(words)
# filtered_word_counts = word_counts - Counter(stop_words)
# print(filtered_word_counts)

# step 3: sort the filtered word counts by frequency
sorted_filtered_counts = sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)

print("Sorted filtered counts:")
print(sorted_filtered_counts)