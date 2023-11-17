# Improting the Modules
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class model:
    
    def __init__(self):
        self.df = pd.read_csv('./spotify_songs.csv')
        self.vectors = None
        self.similarity = None
        
    def readData(self):
        return self.df
    
    def preProcessing(self):
        #Pre Processing the Data - deleteing all the duplicates and the null values

        self.df.drop_duplicates(inplace=True)
        self.df.drop_duplicates(subset=['track_name'], inplace=True)

        self.df.dropna(inplace=True)

        #Pre Processing the data - converting the release data to three separate columns and duration to seconds
        self.df['track_album_release_date'] = pd.to_datetime(self.df['track_album_release_date'], format='mixed')
        self.df['year'] = self.df['track_album_release_date'].dt.year
        self.df['month'] = self.df['track_album_release_date'].dt.month
        self.df['day'] = self.df['track_album_release_date'].dt.day

        self.df['duration_min'] = self.df['duration_ms'] / 60000

        #Pre Processing - removing the unnecssary data
        self.df.drop(columns=['track_id','track_album_id', 'track_album_release_date', 'playlist_id', 'duration_ms'], axis=1, inplace=True)

        #Pre Processing - creating a separate column called desc, based on features provided in Readme
        num_columns = ['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','year','month','day','duration_min']
        for col in num_columns:
            self.df[col] = self.df[col].astype(str)

        self.df['track_artist'] = self.df['track_artist'].str.replace(" ", "")
        self.df['playlist_name'] = self.df['playlist_name'].str.replace(" ","")

        self.df['desc'] = self.df[['track_artist', 'track_album_name', 'playlist_name', 'playlist_genre', 'playlist_subgenre'] + num_columns].apply(lambda x: ' '.join(x), axis=1)
        self.df['desc'] = self.df['desc'].str.lower()

        self.df.drop(columns=['track_artist', 'track_album_name', 'playlist_name', 'playlist_genre', 'playlist_subgenre','track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','year','month','day','duration_min'], inplace=True)
        
    def sparseMatrix(self):
        #Creating Sparse matrix based on the words
        
        cv = CountVectorizer(max_features=5000, stop_words='english')
        self.vectors = cv.fit_transform(self.df['desc'])
        return self.vectors
    
    def distanceMatrix(self):
        #Calculating the cosine similarity index also called the distance
        self.similarity = cosine_similarity(self.vectors, dense_output=False) 
        return self.similarity

