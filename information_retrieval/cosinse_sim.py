import numpy as np


def cosine_similarity_numpy(vec1, vec2):
    """
    Calculates the cosine similarity between two 1D NumPy vectors.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0

    cosine_similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return cosine_similarity


vector1 = np.array([[1, 2, 3]])
vector2 = np.array([[4, 5, 6], [7, 8, 9]]).T
similarity = cosine_similarity_numpy(vector1, vector2)

print(f"Cosine Similarity: {similarity}")

print(np.sort(similarity)[::-1][0])
