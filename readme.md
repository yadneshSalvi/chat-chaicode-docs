## Process followed:
1. Created a sitemap for the website https://docs.chaicode.com/
2. For each page in the sitemap, I crawled the page and extracted the content using crawl4ai.
3. I processed the content to remove the header part which was not relevant to the content.
4. I created vector embeddings for the content and stored them in qdrant.
5. I created a qna pipeline using langchain and qdrant.
6. Created a streamlit app to interact with the qna pipeline.

## To run the streamlit app:
1. Install the dependencies from requirements.txt
2. Create a .env file and add the following variables:
qdrant_api_key=
qdrant_endpoint=
OPENAI_API_KEY=
3. Run the streamlit app using the command: streamlit run streamlit.py

## Features:
1. User can ask questions regarding the documentation.
2. User can ask follow-up questions that reference previous answers.
3. User will get relevant links to the documentation to access the full content.