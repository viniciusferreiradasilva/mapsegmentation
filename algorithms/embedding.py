import numpy as np

"""
File containing embedding strategies to embed districts categories to recomendation.
"""


def embedding_by_category_probability(categories, districts_categories_count):
    """Clustering algorithm based on main attribute.

    Parameters
    ----------

    categories : A python dict containing all the categories in the dataframe.

    districts_categories_count : A python list where each object is a dict representing a district. The keys of the dict
    represent the the categories ids and the values represent the counting of appearances of that category.

    Returns
    -------
    districts_categories_embedding: a python list where each object is a dict representing a district. The keys of the
    dict represent the categories ids and the values represent the probability of such category exist in the district.
    """
    districts_categories_embedding = [None] * len(districts_categories_count)
    for district_id, district in enumerate(districts_categories_count):
        # Calculates the value of embedding for every present category.
        embedding_values = np.divide(list(district.values()), np.sum(list(district.values())))
        # Merges an empty district embedding to the existent district embedding to give zero value to non-present
        # categories.
        district_embedding = {**dict(zip(categories.values(), np.zeros(len(categories)))),
                              **dict(zip(district.keys(), embedding_values))}
        # Add the district embedding to the embedding dictionary.
        districts_categories_embedding[district_id] = district_embedding
    return districts_categories_embedding
