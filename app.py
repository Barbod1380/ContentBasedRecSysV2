import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
import os


def fetch_poster(movies_info, movie_names):
    posters = []
    for movie in movie_names:
        poster_url = movies_info[movies_info['Title'] == movie]['Poster'].values[0]
        posters.append(poster_url)
    return posters


def recommend(titles, movie_name, cosine_mat, count):
    pos = np.where(titles == movie_name)[0][0]
    neighbors = cosine_mat[pos]
    nearest_neighbors = neighbors[-count-1:-1]
    nearest_neighbors = nearest_neighbors[::-1]
    nearest_neighbors_names = titles[nearest_neighbors]
    return nearest_neighbors_names


def fetch_movie_details(movie_title):
    api_key = 'd3f4a2fb'
    url = f'http://www.omdbapi.com/?t={movie_title}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  
    return None


def extract_ratings(ratings):
    rotten_tomatoes = None
    metacritic = None
    for rating in ratings:
        if rating['Source'] == 'Rotten Tomatoes':
            rotten_tomatoes = rating['Value']
        if rating['Source'] == 'Metacritic':
            metacritic = rating['Value']
    return rotten_tomatoes, metacritic


st.markdown("""
    <style>
    img:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    
    div:hover span {
        color: #FFA500;
        transition: 0.2s;
    }

    .stApp {
        background-image: url('https://wallpapercave.com/wp/wp8492327.jpg');
        background-size: cover;
    }

    .movie-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 15px;
        width: 85%;
        margin: auto;
    }

    .movie-poster {
        float: left;
        margin-right: 20px;
    }

    .movie-details {
        color: white;
        font-family: Arial, sans-serif;
        font-size: 18px;
    }

    .rating-logo {
        width: 30px;
        vertical-align: middle;
        margin-right: 10px;
    }

    .rating-box {
        margin: 10px 0;
        display: flex;
        align-items: center;
    }

    .custom-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        text-align: center;
        cursor: pointer;
        margin-bottom: 20px;
    }
    .custom-button:hover {
        background-color: #45a049;
        transition: background-color 0.3s;
    }

    .chat-container {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        height: 75vh;
        overflow-y: auto;
    }

    .user-message {
        background-color: #FFA500;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        width: 80%;
    }

    .bot-response {
        background-color: #333;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        width: 80%;
    }

    .chat-input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        width: 100%;
        margin-bottom: 10px;
    }

    .custom-submit-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 15px;
        font-size: 14px;
        border-radius: 5px;
        cursor: pointer;
    }
    .custom-submit-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html = True)

st.markdown("<h1 style='text-align: center; color: #FFA500;'>Movie Recommendation</h1>", unsafe_allow_html=True)


cosine_matrix = np.load('CosineMatrix50.npy')
movies_info = pd.read_csv('Info.csv').reset_index(drop=True)

movies_title = movies_info.Title.values
selected_movie = st.selectbox('Select movie to get recommendation', movies_title)


os.environ["GROQ_API_KEY"] = "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"


system_prompt = """
Hello! You are a helpful assistant on a movie recommender system website. 
Your role is to answer users' questions about movies, actors, and related topics to help them decide whether to watch a movie. 
Provide accurate, concise, and engaging responses, keeping the tone fun and respectful. 
Include both positive and negative aspects, like reviews or critiques, and describe the mood or themes of the movie to give users a complete picture. 
If you're missing any information, politely ask the user for more details. 
Never fabricate information, and stick to what is known.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=system_prompt
        ),  # This is the persistent system prompt that is always included at the start of the chat.

        MessagesPlaceholder(
            variable_name = "chat_history"
        ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # This template is where the user's current input will be injected into the prompt.
    ]
)

memory = ConversationBufferWindowMemory(k = 10, memory_key = "chat_history", return_messages = True)

llm = ChatGroq(
    model = "llama-3.1-70b-versatile", 
    temperature = 0.0,
)

conversation = LLMChain(
    llm=llm,  # The Groq LangChain chat object initialized earlier.
    prompt=prompt,  # The constructed prompt template.
    verbose=True,   # Enables verbose output, which can be useful for debugging.
    memory=memory,  # The conversational memory object that stores and manages the conversation history.
)

st.sidebar.title("Movie Chat Bot")
st.sidebar.write("Hi, I'm your assistant! You can ask me anything about movies.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    st.session_state['output_respo'] = []

user_input = st.sidebar.text_input(key="user_input", label = "")

if st.sidebar.button("Submit", key = "submit_button"):
    if user_input:
        st.session_state['chat_history'].append(f"<div class='user-message'>{user_input}</div>")

        with st.spinner("Fetching answer..."):
            response = conversation.predict(human_input = user_input)
            message = {'human':user_input, 'AI':response}           
            st.session_state['chat_history'].append(f"<div class='bot-response'>{message}</div>")
            
            st.session_state['output_respo'].append(f"<div class='user-message'>{message['human']}</div>")
            st.session_state['output_respo'].append(f"<div class='bot-response'>{message['AI']}</div>")

# Display chat history
st.sidebar.markdown("<div class='chat-container'>" + "".join(st.session_state['output_respo']) + "</div>", unsafe_allow_html=True)




# Initialize session state for selected movie
if 'current_movie' not in st.session_state:
    st.session_state['current_movie'] = None

# Variable to hold recommended movies and posters
recommended_movies_name = []
posters = []

# Show recommendations when button is clicked
if st.button('Show Recommendation', key='show_rec_button'):
    with st.spinner('Fetching recommendations...'):
        recommended_movies_name = recommend(movies_title, selected_movie, cosine_matrix, 10)
        posters = fetch_poster(movies_info, recommended_movies_name)

# Create buttons for recommended movies
if len(recommended_movies_name) != 0:
    st.subheader("Top Recommendations")
    cols1 = st.columns(5)
    for i, col in enumerate(cols1):
        with col:
            st.button(recommended_movies_name[i][:10] + '...', key=f"button_{i}", on_click=lambda m=recommended_movies_name[i]: st.session_state.update({'current_movie': m}))
            st.image(posters[i], width=120)

    st.subheader("More Recommendations")
    cols2 = st.columns(5)
    for i in range(5, 10):
        with cols2[i - 5]:
            st.button(recommended_movies_name[i][:10] + '...', key=f"button_{i}", on_click = lambda m = recommended_movies_name[i]: st.session_state.update({'current_movie': m}))
            st.image(posters[i], width = 120)

# Display movie details
if st.session_state['current_movie'] is not None:
    movie_details = fetch_movie_details(st.session_state['current_movie'])
    if movie_details and movie_details['Response'] == 'True':
        rotten_tomatoes, metacritic = extract_ratings(movie_details.get('Ratings', []))

        st.markdown(f"""
        <div class='movie-container'>
            <div class='movie-poster'>
                <img src="{movie_details['Poster']}" width="200"/>
            </div>
            <div class='movie-details'>
                <h2>{movie_details['Title']} ({movie_details['Year']})</h2>
                <p><strong>Director:</strong> {movie_details['Director']}</p>
                <p><strong>Actors:</strong> {movie_details['Actors']}</p>
                <p><strong>Plot:</strong> {movie_details['Plot']}</p>
                <div class='rating-box'>
                    <img class='rating-logo' src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg"/>
                    <span>{movie_details['imdbRating']}</span>
                </div>
                <div class='rating-box'>
                    <img class='rating-logo' src="https://upload.wikimedia.org/wikipedia/commons/0/08/FreshTomato.svg"/>
                    <span>{rotten_tomatoes or 'N/A'}</span>
                </div>
                <div class='rating-box'>
                    <img class='rating-logo' src="https://upload.wikimedia.org/wikipedia/commons/2/20/Metacritic.svg"/>
                    <span>{metacritic or 'N/A'}</span>
                </div>
                <div class='rating-box'>
                    <img class='rating-logo' src="https://upload.wikimedia.org/wikipedia/commons/5/52/Ticket.svg"/>
                    <span>{movie_details['BoxOffice'] or 'N/A'}</span>
                </div>
            </div>
            <div style='clear: both'></div>
        </div>
        """, unsafe_allow_html=True)



        