from model import model



def recommend(title, df, similarity_matrix, count_vectorizer):
    title_index = df.index[df['track_name'] == title]
    if len(title_index) == 0:
        print("Song not found.")
        return

    title_index = title_index[0]
    distances = similarity_matrix[title_index]
    
    distances = enumerate(distances.todense().tolist()[0])
    distances = sorted(distances, key = lambda x: x[1], reverse = True)
    
    track_name = df['track_name'].tolist()
    
    for i in range(11):
        index = distances[i][0]
        print(track_name[index])
        
Model = model()
data = Model.readData()
Model.preProcessing()
sparseMatrix = Model.sparseMatrix()
distanceMatrix = Model.distanceMatrix()

while True:
    try:
        title = input("Enter the name of song: ")
        recommend(title, data, distanceMatrix, sparseMatrix)
    except EOFError:
        print("Exiting the program.")
        break
        
    
