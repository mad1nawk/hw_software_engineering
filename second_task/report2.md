Second task:
a)
Problem 1: Cold start for a new user and track
    Problem: The Million Song Dataset contains millions of tracks and users. For a new user, there is no listening history, 
        and for a new track, there are no interactions.
    Possible solution: Use heuristics (global popularity) or request a short survey upon registration (selecting 3–5 
        favorite genres or artists).
    Problem with this solution: The survey increases the registration duration and leads to an increase in early user churn. 
        Popularity‑based recommendations are not personalized.
    Best Practices: Instead of a mandatory survey, use non-invasive onboarding: show popular/trending content 
        by dynamically rearranging the feed immediately after the first 2-3 likes or listenings. For tracks - in the first 
        hours after the release, artificially mixing them into the feeds of the relevant audience.

Problem 2: "Echo Chamber" (Filter Bubble) 
    Problem: Algorithms maximize current retention and start recommending only the same narrow genre to the user. The user 
        is isolated from new music, and the platform is degraded because it cannot develop their taste. 
    Possible solution: Implementation of the "randomness" parameter (serendipity).
    The problem with this solution: The increase in diversity and random recommendations reduces the classic ML accuracy 
        metrics at the moment, the system will offer less obvious options that the user will skip (skip)
    Best Practices: Divide recommendation scenarios into blocks. The "home feed" should exploit the user's current tastes, 
        and special blocks ("Discoveries of the Week" on Spotify) should focus on a controlled exit beyond the bubble

b)
Problem 3: Data and concept drift
    Problem: music trends change quickly. A song that was popular last summer loses its relevance in winter (concept drift),
        and on Friday, 100 new albums are released, completely changing the consumption structure (data drift). If the 
        model is retrained infrequently, the recommendations become irrelevant.
    Possible solution: Continuous and frequent retraining of models (for example, every hour or in real time).
    The problem with this solution: the risk of model instability (“forgetting” the user’s old tastes), as well as enormous
        financial costs for cloud infrastructure.
    Best practices: dividing features into static (genre, long‑term taste profile over a year) and dynamic(trends over the last 2 hours).


Third task:
1.1. ML metrics (Algorithm quality assessment)
    Precision@K and Recall@K: The proportion of relevant recommendations in the top-K issued positions. For example, if 
        a user adds 1 out of 10 recommended tracks to their playlist, Precision@10 = 0.1.
    NDCG (Normalized Discounted Cumulative Gain): Takes into account not only the fact of interaction but also the position 
        of the recommendation. A relevant track in 1st place is valued higher than one in 10th place.
    Coverage (Catalog Coverage): The percentage of the entire track catalog that the system ever recommends. Low coverage 
        indicates that the system is only looping back to “hits” (popularity bias).
    Novelty: Assesses how unexpected (not the most obvious) tracks the system recommends. Helps measure the departure 
        from the “filter bubble.”

1.2. Business metrics (Assessing the impact on the product)
    Skip Rate: The percentage of recommended tracks that the user skipped in the first 30 seconds. A high skip rate is 
        a direct indicator of poor recommendation relevance.
    Listening Time (Listening Time): The average time spent in the app per session. A successful RecSys should increase 
        this metric by retaining user attention.
    Retention Rate (Retention): The percentage of users who return to the app on the 7th or 8th day. High-quality 
        personalization directly affects this metric.
    Conversion to Premium: The share of users who have switched to a paid subscription. Good recommendations prove 
        the value of the paid service.
