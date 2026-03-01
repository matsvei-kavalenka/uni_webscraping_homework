import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorise_data(df_col):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
    )

    vec = vectorizer.fit_transform(df_col)

    word_counts = pd.DataFrame(
        vec.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    total_word_counts = word_counts.sum().sort_values(ascending=False)

    return total_word_counts

