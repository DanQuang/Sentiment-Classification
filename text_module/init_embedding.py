from text_module.count_vectorizer import CountVectorizer
from text_module.word_embedding import WordEmbedding

def build_text_embedding(config):
    if config["text_embedding"]["type_embedding"] == "count_vector":
        return CountVectorizer(config)
    if config["text_embedding"]["type_embedding"] == "word_embedding":
        return WordEmbedding(config)