import streamlit as st
import pandas as pd


# Available models
models = (
    "Course Similarity",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "KNN",
    "NMF",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features"
)


# Loaders
def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    df['COURSE_ID'] = df['COURSE_ID'].str.upper()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


# Add new user ratings
def add_new_ratings(new_courses):
    if len(new_courses) == 0:
        return None
    ratings_df = load_ratings()
    new_id = ratings_df['user'].max() + 1
    new_df = pd.DataFrame({
        'user': [new_id] * len(new_courses),
        'item': [cid.upper() for cid in new_courses],
        'rating': [3.0] * len(new_courses)
    })
    updated_ratings = pd.concat([ratings_df, new_df], ignore_index=True)
    updated_ratings.to_csv("ratings.csv", index=False)
    return new_id



# BOW dictionary
def get_doc_dicts():
    bow_df = load_bow()
    bow_df['doc_id'] = bow_df['doc_id'].str.upper()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    return idx_id_dict, id_idx_dict


# Course similarity scoring
def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    enrolled_course_ids = [cid.upper() for cid in enrolled_course_ids]
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselected_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselected_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselected_course]
                sim = sim_matrix[idx1][idx2]
                if unselected_course not in res or sim > res[unselected_course]:
                    res[unselected_course] = sim
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    return res


# Model training
def train(model_name, params):
    if model_name == models[0]:  # "Course Similarity"
        if 'enrolled_course_ids' in params:
            idx_id_dict, id_idx_dict = get_doc_dicts()
            course_df = load_courses()
            enrolled_course_ids = [cid.upper() for cid in params['enrolled_course_ids']]
            enrolled_courses = course_df[course_df['COURSE_ID'].isin(enrolled_course_ids)]
            st.write(f"Training for Course Similarity Model with {len(enrolled_course_ids)} enrolled courses.")
            st.dataframe(enrolled_courses[['COURSE_ID', 'TITLE']])
        else:
            st.error("Enrolled courses not provided in params.")
    elif model_name in models:
        st.write(f"Training for {model_name} model.")
    else:
        st.error(f"Model {model_name} not recognized.")


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = params.get("sim_threshold", 60) / 100.0

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        if model_name == models[0]:  # Course Similarity
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = [cid.upper() for cid in user_ratings['item'].to_list()]
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

    res_df = pd.DataFrame({
        'USER': users,
        'COURSE_ID': courses,
        'SCORE': scores
    })

    if not res_df.empty:
        courses_df = load_courses()
        final_df = pd.merge(res_df, courses_df, on='COURSE_ID', how='left')
        final_df = final_df[['SCORE', 'TITLE', 'DESCRIPTION']]
        return final_df
    else:
        return pd.DataFrame(columns=["SCORE", "TITLE", "DESCRIPTION"])
