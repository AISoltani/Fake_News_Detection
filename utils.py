import pickle

def load_vectorizer():
    with open("Vectorizer.pkl","rb") as f:
                vect = pickle.load(f)
    return vect

def text_with_title(text):
        res = text['title'] + " " + text["text"]
        return res
