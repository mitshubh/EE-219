import numpy as np

hour_status = []
celeb_first = ["John", "Idina", "Katy", "Lenny", "Missy", "Nina", "Josh"]
celeb_last = ["Legend", "Menzel", "Perry", "Kravitz", "Elliott", "Dobrev", "Duhamel"]

def get_celebrities(hash_tags, key_words):
    local_celeb_count = 0
    celeb_count = np.zeros(len(celeb_first))

    for count, tweet in enumerate(key_words.keys()):
        for i in range(len(celeb_count)):
            if tweet.find(celeb_first[i].lower()) > -1 or tweet.find(celeb_last[i].lower()) > -1:
                celeb_count[i] += key_words.get(tweet)
                local_celeb_count += key_words.get(tweet)

    for count, tweet in enumerate(hash_tags.keys()):
        for i in range(len(celeb_count)):
            if tweet.find(celeb_first[i].lower()) > -1 or tweet.find(celeb_last[i].lower()) > -1:
                celeb_count[i] += hash_tags.get(tweet)
                local_celeb_count += hash_tags.get(tweet)

    hour_status.append(celeb_count)
    return local_celeb_count