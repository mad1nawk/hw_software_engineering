import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

plays_df = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])
tracks_df = pd.read_csv('unique_tracks.txt', sep='<SEP>', header=None, names=['track_id', 'song_id', 'artist_name', 'title'])
df = pd.merge(plays_df, tracks_df, on='song_id', how='left')


def item_based_collaborative_filtering(user_id, df, top_n=2):
    user_history = df[df['user_id'] == user_id]
    if user_history.empty:
        return "Недостаточно данных для коллаборативной фильтрации."

    user_item_matrix = df.pivot_table(index='user_id', columns='song_id', values='play_count', fill_value=0)

    item_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    user_songs = user_history['song_id'].unique()
    candidate_songs = [s for s in similarity_df.columns if s not in user_songs]

    recommendations = []
    for song in candidate_songs:
        score = sum(similarity_df[song][user_song] for user_song in user_songs)
        recommendations.append((song, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_songs = [rec[0] for rec in recommendations[:top_n]]

    result = df[df['song_id'].isin(top_songs)][['song_id', 'title', 'artist_name']]
    return result.drop_duplicates().to_string(index=False)

def content_based_filtering(target_song_id, df, top_n=2):
    df['content_features'] = df['artist_name'] + " " + df['genre']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['content_features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    if target_song_id not in df['song_id'].values:
        return "Целевая песня не найдена в датасете."
        
    idx = df[df['song_id'] == target_song_id].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    result = df.iloc[top_indices][['song_id', 'title', 'artist_name', 'genre']]
    return result.to_string(index=False)
