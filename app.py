import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data  # Updated to use @st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data  # Updated to use @st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data  # Updated to use @st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data  # Updated to use @st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results

def train(model_name, params):
    if model_name == backend.models[0]:  # Course Similarity
        with st.spinner('Training Course Similarity model...'):
            time.sleep(0.5)
            backend.train(model_name, params)  # Pass both model_name and params
        st.success('Course Similarity model training complete!')
    elif model_name == backend.models[1]:  # User Profile
        with st.spinner('Training User Profile model...'):
            time.sleep(0.5)
            backend.train(model_name, params)  # Pass both model_name and params
        st.success('User Profile model training complete!')
    # Handle other models similarly
    else:
        pass


def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0

    idx_id_dict, id_idx_dict = backend.get_doc_dicts()
    sim_matrix = backend.load_course_sims().to_numpy()
    users, courses, scores = [], [], []
    res_dict = {}

    for user_id in user_ids:
        if model_name == backend.models[0]:  # Course Similarity
            ratings_df = backend.load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = [cid.upper() for cid in user_ratings['item'].to_list()]
            res = backend.course_similarity_recommendations(
                idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix
            )
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses  # Make sure COURSE_ID is included
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict)

    if not res_df.empty:
        courses_df = backend.load_courses()
        final_df = pd.merge(res_df, courses_df, on='COURSE_ID', how='left')
        final_df = final_df[['COURSE_ID', 'SCORE', 'TITLE', 'DESCRIPTION']]  # Keep COURSE_ID
        return final_df
    else:
        return pd.DataFrame(columns=["COURSE_ID", "SCORE", "TITLE", "DESCRIPTION"])



# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')

# Start training process
if training_button:
    if not selected_courses_df.empty:
        params['enrolled_course_ids'] = selected_courses_df['COURSE_ID'].tolist()
    train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df)
