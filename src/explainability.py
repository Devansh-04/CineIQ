def explain_content_similarity(movie_title, recommended_title, content_score):
    return (
        f"'{recommended_title}' is recommended because it is content-wise "
        f"similar to '{movie_title}'. The content similarity score is "
        f"{content_score:.2f}."
    )


def explain_user_preference(user_preference_score):
    if user_preference_score >= 0.8:
        return "It strongly matches the user's preferred genres."
    elif user_preference_score >= 0.5:
        return "It moderately matches the user's preferred genres."
    elif user_preference_score > 0:
        return "It has a small match with the user's genre preferences."
    else:
        return "It does not strongly match the user's previous genre preferences."


def explain_rating_signal(avg_rating_score):
    if avg_rating_score >= 0.8:
        return "It has strong average rating signals from users."
    elif avg_rating_score >= 0.6:
        return "It has decent average rating signals from users."
    elif avg_rating_score > 0:
        return "It has some rating signal, but it is not very strong."
    else:
        return "There is not enough rating signal available."


def explain_popularity(popularity_score):
    if popularity_score >= 0.8:
        return "It is a very popular movie among users."
    elif popularity_score >= 0.5:
        return "It has moderate popularity among users."
    elif popularity_score > 0:
        return "It has limited popularity but may still be relevant."
    else:
        return "It has very low popularity data available."


def explain_sentiment(sentiment_score):
    if sentiment_score >= 0.75:
        return "Audience sentiment signal is positive."
    elif sentiment_score >= 0.5:
        return "Audience sentiment signal is neutral to slightly positive."
    elif sentiment_score > 0:
        return "Audience sentiment signal is weak or slightly negative."
    else:
        return "No useful sentiment signal is available."


def generate_explanation(
    input_movie_title,
    recommended_movie_title,
    content_score,
    user_preference_score,
    avg_rating_score,
    popularity_score,
    sentiment_score
):
    """
    Generates human-readable explanation for recommendation.
    """

    explanation_parts = [
        explain_content_similarity(
            input_movie_title,
            recommended_movie_title,
            content_score
        ),
        explain_user_preference(user_preference_score),
        explain_rating_signal(avg_rating_score),
        explain_popularity(popularity_score),
        explain_sentiment(sentiment_score)
    ]

    return " ".join(explanation_parts)


def explain_recommendation_row(row, input_movie_title):
    """
    Takes one recommendation row and returns explanation.
    """

    recommended_title = row.get("title", "This movie")

    content_score = row.get("content_score", 0)
    user_preference_score = row.get("user_preference_score", 0)
    avg_rating_score = row.get("avg_rating_score", 0)
    popularity_score = row.get("popularity_score", 0)
    sentiment_score = row.get("sentiment_score", 0)

    return generate_explanation(
        input_movie_title=input_movie_title,
        recommended_movie_title=recommended_title,
        content_score=content_score,
        user_preference_score=user_preference_score,
        avg_rating_score=avg_rating_score,
        popularity_score=popularity_score,
        sentiment_score=sentiment_score
    )