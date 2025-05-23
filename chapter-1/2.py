from diversity.patterns.part_of_speech import pos_patterns
from collections import Counter # Needed for frequency counting
import ssl # Needed for potential NLTK SSL fix
import nltk # Needed for NLTK download/fix


# --- START: Attempt to fix NLTK SSL certificate issues and download data ---
# This block tries to bypass SSL verification for NLTK downloads and then
# attempts to download the necessary data. It MUST run before pos_patterns.
try:
    # Attempt to bypass SSL verification for NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python versions might not have this
        pass
    else:
        # Apply the bypass - use with caution, primarily for isolated scripts
        ssl._create_default_https_context = _create_unverified_https_context

    # Now attempt to download the NLTK data required by pos_patterns
    # The previous Index Error suggests 'averaged_perceptron_tagger' and likely 'punkt' are needed.
    print("Attempting to download NLTK data (averaged_perceptron_tagger, punkt)...")
    # Using quiet=False temporarily to see download messages.
    # force=False prevents unnecessary redownload if already present.
    nltk.download('averaged_perceptron_tagger', quiet=False, force=False)
    nltk.download('punkt', quiet=False, force=False)
    print("NLTK data download attempt finished.")

    # If the IndexError persists, you might need to run nltk.download() interactively
    # (just 'python' then 'import nltk; nltk.download()') and try downloading 'all'
    # or components like 'wordnet', 'omw-1.4'. Observe the NLTK downloader GUI.

except Exception as e:
    print(f"Warning: Error during NLTK data download attempt: {e}. Pattern extraction might be affected.")
# --- END: Attempt to fix NLTK SSL certificate issues and download data ---


human_essay = """
A software developer, also known as a developer or programmer, uses programming skills to create new software and improve existing applications. They design and write the code used to build everything from operating systems to mobile apps and video games. The developer role involves every stage of the software development life cycle, from understanding user needs to releasing a completed application.
"""
llm_essay = """
Developers are the architects of the digital world, building the software, websites, and applications we use daily. Through code and problem-solving, they transform concepts into tangible realities, driving innovation and shaping our connected lives. Their expertise is fundamental to progress, constantly pushing technological boundaries and creating the tools that define the modern era.
"""
n = 5 # Pattern length (n-gram of POS tags)
# Removed top_n from here as it's not accepted by pos_patterns function call directly


print(f"\nExtracting POS patterns with n={n} from Human essay...")
# Call the function with ONLY 2 arguments: list of text and n
try:
    # Pass text in a list as expected by the library
    # Based on docs and inspection, it returns a dictionary {pattern: [list of text examples]}
    human_raw_patterns = pos_patterns([human_essay], n) # Removed top_n here
    # Get the list of patterns (the keys of the returned dictionary) to count frequencies
    human_pattern_list = list(human_raw_patterns.keys())
except Exception as e:
    print(f"Error during human pattern extraction: {e}")
    human_pattern_list = [] # Set to empty list to avoid errors later


print(f"\nExtracting POS patterns with n={n} from LLM essay...")
# Call the function with ONLY 2 arguments: list of text and n
try:
    llm_raw_patterns = pos_patterns([llm_essay], n) # Removed top_n here
    # Get the list of patterns (the keys of the returned dictionary)
    llm_pattern_list = list(llm_raw_patterns.keys())
except Exception as e:
    print(f"Error during LLM pattern extraction: {e}")
    llm_pattern_list = [] # Set to empty list


print("\n--- Frequency Analysis ---")

top_n_to_print = 10 # How many of the most frequent patterns *from our analysis* to print

if human_pattern_list:
    human_pattern_counts = Counter(human_pattern_list)
    most_common_human = human_pattern_counts.most_common(top_n_to_print)
    print("\nMost common Human POS patterns:")
    # The 'count' here is how many times each distinct pattern type was found
    # as a key in the dictionary returned by pos_patterns.
    # The relative frequency is relative to the total number of distinct pattern types found.
    print(f"Total distinct pattern types found: {len(human_pattern_list)}")
    for pattern, count in most_common_human:
        relative_freq = count / len(human_pattern_list) if human_pattern_list else 0
        print(f"  '{pattern}': {count} appearance(s) among distinct patterns ({relative_freq:.2%})")
else:
    print("\nNo human patterns extracted or analyzed.")

if llm_pattern_list:
    llm_pattern_counts = Counter(llm_pattern_list)
    most_common_llm = llm_pattern_counts.most_common(top_n_to_print)
    print("\nMost common LLM POS patterns:")
    print(f"Total distinct pattern types found: {len(llm_pattern_list)}")
    for pattern, count in most_common_llm:
        relative_freq = count / len(llm_pattern_list) if llm_pattern_list else 0
        print(f"  '{pattern}': {count} appearance(s) among distinct patterns ({relative_freq:.2%})")
else:
     print("\nNo LLM patterns extracted or analyzed.")

print("\n--- Comparison ---")
print("Compare the lists of most common POS patterns and their frequencies:")
print("- Are the most frequent POS patterns similar or different?")
print("- How do their distinct appearance counts or relative frequencies compare?")
print("\nKeep in mind the NLTK SSL download error might still be affecting the quality or completeness of the extracted patterns if the download failed.")
