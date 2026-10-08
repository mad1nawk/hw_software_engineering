import pandas as pd
import numpy as np

# Указываем engine='python', чтобы скрыть ParserWarning из-за сепаратора '<SEP>'
plays_df = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])
tracks_df = pd.read_csv('unique_tracks.txt', sep='<SEP>', header=None, names=['track_id', 'song_id', 'artist_name', 'title'])
df = pd.merge(plays_df, tracks_df, on='song_id', how='left')


def get_popular_songs(df, top_n=3):
    """Эвристика 1: Глобальная популярность по суммарному количеству прослушиваний."""
    popular = df.groupby(['song_id', 'title', 'artist_name'])['play_count'].sum().reset_index()
    popular = popular.sort_values(by='play_count', ascending=False).head(top_n)
    return popular[['title', 'artist_name', 'play_count']].to_string(index=False)


def get_same_artist_recommendations(user_id, df, top_n=2):
    """Эвристика 2: Рекомендация других песен того же исполнителя, которого слушал пользователь."""
    user_history = df[df['user_id'] == user_id]
    if user_history.empty:
        return "История прослушиваний пуста."

    # Находим самого частого исполнителя пользователя
    top_artist = user_history.groupby('artist_name')['play_count'].sum().idxmax()
    user_songs = user_history['song_id'].unique()

    # Ищем другие песни этого исполнителя
    candidates = df[(df['artist_name'] == top_artist) & (~df['song_id'].isin(user_songs))]

    if candidates.empty:
        return f"Пользователь уже слушал все доступные треки исполнителя {top_artist}."

    return candidates[['title', 'artist_name']].drop_duplicates().head(top_n).to_string(index=False)


print("\n=== Эвристика 1: Глобально популярные треки ===")
print(get_popular_songs(df))

print("\n=== Эвристика 2: Другие треки того же исполнителя (для пользователя u1) ===")
print(get_same_artist_recommendations('u1', df))
