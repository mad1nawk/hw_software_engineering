import pandas as pd

plays_df = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])
tracks_df = pd.read_csv('unique_tracks.txt', sep='<SEP>', header=None, names=['track_id', 'song_id', 'artist_name', 'title'])
df = pd.merge(plays_df, tracks_df, on='song_id', how='left')


def get_popular_songs(df, top_n=3):
    popular = df.groupby(['song_id', 'title', 'artist_name'])['play_count'].sum().reset_index()
    popular = popular.sort_values(by='play_count', ascending=False).head(top_n)
    return popular[['title', 'artist_name', 'play_count']].to_string(index=False)


def get_same_artist_recommendations(user_id, df, top_n=2):
    user_history = df[df['user_id'] == user_id]
    if user_history.empty:
        return "There is not enough data for collaborative filtering"

    top_artist = user_history.groupby('artist_name')['play_count'].sum().idxmax()
    user_songs = user_history['song_id'].unique()
    candidates = df[(df['artist_name'] == top_artist) & (~df['song_id'].isin(user_songs))]

    return candidates[['title', 'artist_name']].drop_duplicates().head(top_n).to_string(index=False)


