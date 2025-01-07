def find_matching_words(word_list, pattern):
    import re

    regex_pattern = pattern.lower().replace("_", ".").replace(" ", "")
    return [word for word in word_list if re.fullmatch(regex_pattern, word)]


def calculate_shared_letter_score(word, word_list):
    score = 0
    for other_word in word_list:
        if word != other_word:
            score += sum(1 for a, b in zip(word, other_word) if a == b)
    return score


def find_word_with_most_info(word_list):
    shared_scores = {
        word: calculate_shared_letter_score(word, word_list) for word in word_list
    }
    return max(shared_scores, key=shared_scores.get), shared_scores


# Reload the file content
file_path = "pilgram/data/words.txt"
with open(file_path, "r") as file:
    words = file.read().splitlines()

# Define the pattern for "____s___"
pattern = "_ou__"

# Find matching words
potential_words = find_matching_words(words, pattern)
if not potential_words:
    print("No matching words found.")
else:
    word_with_most_info, shared_scores = find_word_with_most_info(potential_words)
    print(f"Word with the most shared letters: {word_with_most_info}")
    print(f"Shared letter scores {shared_scores}")
