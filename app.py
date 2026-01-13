import streamlit as st
import spacy
import google.generativeai as genai
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
import couchbase.search as search
from couchbase.vector_search import VectorQuery, VectorSearch
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="History RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION (FROM ENVIRONMENT OR SIDEBAR) ---
st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("API & Database Settings", expanded=False):
    gemini_api_key = st. text_input(
        "Gemini API Key",
        type="password",
        value="YOUR_GEMINI_API_KEY"
    )
    cb_url = st.text_input(
        "Couchbase URL",
        value="http://127.0.0.1:8091"
    )
    cb_user = st.text_input(
        "Couchbase Username",
        value="Administrator"
    )
    cb_pass = st.text_input(
        "Couchbase Password",
        type="password",
        value="poweRsub66"
    )
    bucket_name = st.text_input(
        "Bucket Name",
        value="h"
    )
    index_name = st.text_input(
        "Vector Index Name",
        value="vector_index"
    )

# --- INITIALIZE SESSION STATE ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.cluster = None
    st.session_state.bucket = None
    st.session_state.scope = None
    st.session_state.nlp = None
    st.session_state.llm = None
    st.session_state.conversation_history = []

# --- INITIALIZE AI AND DATABASE ---
def initialize_services():
    try:
        with st.spinner("Initializing services..."):
            # Initialize Gemini
            genai.configure(api_key=gemini_api_key)
            llm = genai.GenerativeModel('gemini-1.5-flash')
            
            # Initialize spaCy
            nlp = spacy.load("en_core_web_md")
            
            # Initialize Couchbase
            auth = PasswordAuthenticator(cb_user, cb_pass)
            cluster = Cluster(cb_url, ClusterOptions(auth))
            bucket = cluster.bucket(bucket_name)
            scope = bucket.scope("_default")
            
            st.session_state.llm = llm
            st. session_state.nlp = nlp
            st.session_state.cluster = cluster
            st. session_state.bucket = bucket
            st.session_state.scope = scope
            st.session_state.initialized = True
            
        st.sidebar.success("✅ All services initialized successfully!")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Initialization failed: {str(e)}")
        return False

# Initialize button
if st.sidebar.button("🔄 Initialize Services", key="init_btn"):
    initialize_services()

if st.session_state.initialized:
    st.sidebar.success("✅ Services Ready")
else:
    st.sidebar.warning("⚠️ Click 'Initialize Services' to begin")

# --- MAIN INTERFACE ---
st.title("📚 History RAG Assistant")
st.markdown("Ask questions about history in Hindi or English and get answers powered by RAG")

# --- SIDEBAR OPTIONS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Options")
show_context = st.sidebar.checkbox("Show Retrieved Context", value=True)
num_candidates = st.sidebar.slider("Number of Search Results", 1, 10, 3)
language_mode = st.sidebar.radio(
    "Query Language",
    ["Hindi", "English", "Auto-detect"]
)

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state. conversation_history = []
    st.rerun()

# --- RAG FUNCTION ---
def run_history_rag(hindi_prompt, language_mode="Auto-detect"):
    """
    Run the RAG pipeline with Couchbase vector search
    """
    try:
        # Step 1: Language Detection & Translation
        st.info("🔍 Processing query...")
        
        if language_mode == "Auto-detect":
            # Detect if input is Hindi
            is_hindi = any('\u0900' <= char <= '\u097F' for char in hindi_prompt)
        else:
            is_hindi = language_mode == "Hindi"
        
        if is_hindi: 
            trans_task = f"Translate this Hindi query to English for a history search: {hindi_prompt}"
            eng_query = st.session_state.llm.generate_content(trans_task).text.strip()
            st.success(f"✅ Translated to:  {eng_query}")
        else:
            eng_query = hindi_prompt
            st.success("✅ Using English query directly")
        
        # Step 2: spaCy Enrichment & Vector Generation
        st.info("🧠 Enriching query with NLP...")
        doc = st.session_state.nlp(eng_query)
        clean_query = " ".join([token. lemma_ for token in doc if not token.is_stop])
        query_vector = st.session_state.nlp(clean_query).vector.tolist()
        st.success(f"✅ Query enriched:  {clean_query}")
        
        # Step 3: Couchbase Vector Search
        st. info(f"🔎 Searching vector index '{index_name}'...")
        
        search_req = search.SearchRequest. create(
            VectorSearch. from_vector_query(
                VectorQuery("vector_field", query_vector, num_candidates=num_candidates)
            )
        )
        
        retrieved_docs = []
        retrieved_text = ""
        
        try:
            results = st.session_state.scope.search(index_name, search_req)
            for row in results:
                try:
                    doc_content = st.session_state.scope. collection("_default").get(row.id).content_as[dict]
                    content = doc_content.get('content', '')
                    retrieved_docs.append({
                        'id': row. id,
                        'content':  content,
                        'score': row.score if hasattr(row, 'score') else 'N/A'
                    })
                    retrieved_text += content + "\n"
                except Exception as e:
                    st.warning(f"⚠️ Could not retrieve document {row.id}: {str(e)}")
            
            st.success(f"✅ Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            st.error("Ensure the vector index is created in the Couchbase Admin Console")
            retrieved_text = "No documents retrieved"
        
        # Step 4: Generate Response in Hindi
        st.info("🤖 Generating response...")
        
        final_prompt = f"""
Based on the following context from historical documents, answer the question in Hindi.

CONTEXT: 
{retrieved_text}

QUESTION (English): {eng_query}
ORIGINAL QUESTION (Hindi): {hindi_prompt}

Please provide a comprehensive answer in Hindi, referencing the context provided. 
"""
        
        response_text = ""
        response_placeholder = st.empty()
        
        # Stream the response
        response = st.session_state.llm. generate_content(final_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                response_text += chunk.text
                response_placeholder.markdown(response_text)
        
        return {
            'query': hindi_prompt,
            'translated_query': eng_query,
            'response':  response_text,
            'retrieved_docs': retrieved_docs,
            'timestamp': time.time()
        }
    
    except Exception as e: 
        st.error(f"❌ Error in RAG pipeline: {str(e)}")
        return None

# --- MAIN CONTENT ---
if st.session_state.initialized:
    # Input section
    st.subheader("🎯 Ask a Question")
    
    col1, col2 = st. columns([4, 1])
    with col1:
        user_query = st.text_area(
            "Enter your question (in Hindi or English):",
            placeholder="उदाहरण: इब्न बतूता ने डाक व्यवस्था के बारे में क्या बताया?",
            height=100,
            key="user_query"
        )
    
    with col2:
        st.markdown("")  # Spacing
        submit_button = st.button("🔍 Search", use_container_width=True, key="submit_btn")
    
    # Process query
    if submit_button and user_query.strip():
        st.divider()
        
        # Run RAG pipeline
        result = run_history_rag(user_query, language_mode)
        
        if result:
            # Add to conversation history
            st.session_state.conversation_history.append(result)
            
            # Display response
            st.subheader("📖 Response")
            st.markdown(result['response'])
            
            # Display context if enabled
            if show_context and result['retrieved_docs']:
                st.divider()
                st.subheader("📚 Retrieved Context")
                
                tabs = st.tabs([f"Document {i+1}" for i in range(len(result['retrieved_docs']))])
                for idx, (tab, doc) in enumerate(zip(tabs, result['retrieved_docs'])):
                    with tab: 
                        st.write(f"**Document ID:** {doc['id']}")
                        if doc['score'] != 'N/A':
                            st.write(f"**Relevance Score:** {doc['score']:.4f}")
                        st.text_area(
                            "Content:",
                            value=doc['content'],
                            height=200,
                            disabled=True,
                            key=f"doc_{idx}"
                        )
    
    # Conversation history
    if st.session_state.conversation_history:
        st.divider()
        st.subheader("💬 Conversation History")
        
        for idx, item in enumerate(reversed(st.session_state.conversation_history)):
            with st. expander(f"Q{len(st.session_state.conversation_history)-idx}: {item['query'][: 50]}..."):
                st. write(f"**Original Query:** {item['query']}")
                st.write(f"**Translated Query:** {item['translated_query']}")
                st. write(f"**Response:** {item['response']}")
                st.write(f"**Retrieved Documents:** {len(item['retrieved_docs'])}")

else:
    st.warning("⚠️ Please initialize services using the sidebar configuration first.")
    st.info("""
    Steps to get started:
    1. Add your Gemini API key in the Configuration section
    2. Configure Couchbase connection details
    3. Click 'Initialize Services'
    4. Start asking questions in Hindi or English! 
    """)
