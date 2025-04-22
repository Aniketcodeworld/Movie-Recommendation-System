import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Constants --------------------
API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Image"

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide", page_icon="🎬")

# Custom CSS for styling
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        .title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            color: #FF4B4B;
            margin-bottom: 0.5rem;
            padding-top: 1rem;
        }
        .desc {
            text-align: center;
            color: #808495;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        .movie-card {
            border-radius: 10px;
            padding: 0;
            transition: transform 0.2s;
            background: #0E1117;
            border: 1px solid #2E4053;
        }
        .movie-card:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
        }
        .movie-title {
            font-weight: 600;
            color: white;
            text-align: center;
            margin-top: 0.5rem;
            height: 3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("<div class='title'>🎬   Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Discover your next favorite movie with AI-powered recommendations</div>", unsafe_allow_html=True)

# -------------------- TMDb API Functions --------------------
@st.cache_data(show_spinner=False)
def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

@st.cache_data(show_spinner=False)
def get_genre_dict():
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    genres = response.json().get("genres", [])
    return {g["id"]: g["name"] for g in genres}

def is_valid_image(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except:
        return False

# -------------------- Recommendation Logic --------------------
def hybrid_recommend(title, alpha=0.5, year_range=(1950, 2025), min_rating=0):
    results = search_movie(title)
    if not results:
        return [], None

    target = results[0]
    all_movies = results[:20]  # Get more results for better recommendations

    filtered = []
    for m in all_movies:
        poster_path = m.get('poster_path')
        if not poster_path:
            continue

        poster_url = f"{IMAGE_BASE_URL}{poster_path}"
        if not is_valid_image(poster_url):
            continue

        release_year = int(m['release_date'][:4]) if m.get('release_date') else None
        if release_year and not (year_range[0] <= release_year <= year_range[1]):
            continue
        if m.get('vote_average', 0) < min_rating:
            continue
        filtered.append(m)

    if len(filtered) < 2:
        return [], None

    titles = [m['title'] for m in filtered]
    overviews = [m.get('overview', '') for m in filtered]
    votes = [m.get('vote_average', 0) for m in filtered]

    genre_dict = get_genre_dict()

    movie_data = []
    for m in filtered:
        path = m.get('poster_path')
        poster = f"{IMAGE_BASE_URL}{path}"
        genre_ids = m.get('genre_ids', [])
        genres = [genre_dict.get(gid, "Unknown") for gid in genre_ids]

        movie_data.append({
            'title': m['title'],
            'poster': poster,
            'year': m['release_date'][:4] if m.get('release_date') else 'N/A',
            'rating': m.get('vote_average', 0),
            'genres': genres,
            'id': m['id']
        })

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(overviews)
    content_sim = cosine_similarity(tfidf_matrix)

    idx = 0
    content_scores = content_sim[idx]

    vote_scores = pd.Series(votes).fillna(0)
    vote_scores = (vote_scores - vote_scores.min()) / (vote_scores.max() - vote_scores.min())

    final_scores = alpha * content_scores + (1 - alpha) * vote_scores
    sorted_idx = final_scores.argsort()[::-1][1:6]

    recommendations = [movie_data[i] for i in sorted_idx]
    return recommendations, movie_data[0]

# -------------------- Layout --------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🔍 Search for a Movie")
    movie_input = st.text_input("Enter a movie title:", 
                                placeholder="e.g., The Dark Knight, Interstellar...")

    if movie_input:
        with st.spinner('Finding recommendations...'):
            recommendations, target_movie = hybrid_recommend(
                movie_input,
                alpha=st.session_state.get("alpha", 0.5),
                year_range=st.session_state.get("year_range", (2000, 2025)),
                min_rating=st.session_state.get("min_rating", 5.0)
            )

        if recommendations and target_movie:
            st.markdown(f"### 🎯 You searched for: **{target_movie['title']}**")

            cols = st.columns([1, 3])
            with cols[0]:
                st.image(target_movie['poster'], width=150)
            with cols[1]:
                st.markdown(f"""
                    <div style='margin-top: 1rem;'>
                        <h3>{target_movie['title']} ({target_movie['year']})</h3>
                        <p>⭐ <strong>{target_movie['rating']}/10</strong></p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### 🎬 Recommended Movies")
            rec_cols = st.columns(5)
            for idx, movie in enumerate(recommendations):
                with rec_cols[idx % 5]:
                    st.markdown(f"""
                        <div class='movie-card'>
                            <img src='{movie['poster']}' style='width:100%; border-radius: 8px 8px 0 0;'/>
                            <div class='movie-title'>
                                {movie['title']} ({movie['year']})
                            </div>
                            <div style='text-align:center; color:#FF4B4B; padding-bottom:0.5rem;'>
                                ⭐ {movie['rating']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # Export to CSV
            export_df = pd.DataFrame([{
                "Title": m["title"],
                "Year": m["year"],
                "Rating": m["rating"],
                "Genres": ", ".join(m["genres"])
            } for m in recommendations])

            st.download_button(
                label="📅 Download CSV for Power BI",
                data=export_df.to_csv(index=False),
                file_name="movie_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.warning("No recommendations found. Try adjusting filters or search another movie.")

with col2:
    with st.expander("Filter Options", expanded=True):
        st.markdown("**📅 Release Year**")
        year_range = st.slider("Select year range:", 
                               1950, 2025, (2000, 2025),
                               key="year_range",
                               label_visibility="collapsed")

        st.markdown("**⭐ Minimum Rating**")
        min_rating = st.slider("Set minimum rating:", 
                               0.0, 10.0, 5.0, 0.5,
                               key="min_rating",
                               label_visibility="collapsed")

        st.markdown("**⚖️ Recommendation Balance**")
        st.caption("Content-Based Filtering vs Collaborative Filtering")
        alpha = st.slider("Adjust recommendation balance:", 
                          0.0, 1.0, 0.5, 0.1,
                          key="alpha",
                          label_visibility="collapsed")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #808495;'>Powered by TMDb API and Streamlit</div>", unsafe_allow_html=True)
