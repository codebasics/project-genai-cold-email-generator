import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Mail Generator", page_icon="📧")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .title {
            color: #1E90FF;
            font-size: 2em;
        }
        .subtitle {
            color: #4682B4;
            font-size: 1.5em;
        }
        .warning {
            color: #FF6347;
        }
        .success {
            color: #32CD32;
        }
        .sidebar .sidebar-content {
            background-color: #F0F8FF;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">Cold Mail Generator for Job Descriptions</p>', unsafe_allow_html=True)
    st.write("Hi, how are you! I can help to generate a Professional Email!")
    st.write("Generate a tailored cold email based on the job description provided in the URL.")

    st.sidebar.header("Input Window")
    url_input = st.sidebar.text_input("Please Enter a URL:", value="URL please..!")
    submit_button = st.sidebar.button("Generate Email")

    if submit_button:
        if url_input and url_input != "URL please..!":
            try:
                with st.spinner("Processing..."):
                    # Load and clean the job description
                    loader = WebBaseLoader([url_input])
                    raw_data = loader.load().pop().page_content
                    cleaned_data = clean_text(raw_data)
                    
                    # Process the cleaned data and generate email
                    portfolio.load_portfolio()
                    jobs = llm.extract_jobs(cleaned_data)
                    
                    if jobs:
                        for job in jobs:
                            skills = job.get('skills', [])
                            links = portfolio.query_links(skills)
                            email = llm.write_mail(job, links)
                            
                            # Display the generated email
                            st.subheader("Generated Email")
                            st.code(email, language='markdown')
                            st.markdown('<p class="success">Email generated successfully!</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="warning">No jobs found in the provided URL.</p>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<p class="warning">An error occurred: {e}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="warning">Please enter a valid URL.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
