from preprocessing.embeddings import embed

def test_embed():

    embeddings = embed(['hello world'])
    assert embeddings.shape == (1, 768)
