import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Custom CSS for better appearance
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 200px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-box {
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    .neutral {
        background-color: #e2e3e5;
        color: #383d41;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("finetuned_finbert")
        model = AutoModelForSequenceClassification.from_pretrained("finetuned_finbert")
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def get_sentiment_class(sentiment):
    sentiment = sentiment.lower()
    if 'positive' in sentiment:
        return 'positive'
    elif 'negative' in sentiment:
        return 'negative'
    return 'neutral'

def main():
    st.title("📈 Financial Sentiment Analysis")
    st.markdown("Analyze sentiment in financial news using FinBERT")
    
    classifier = load_model()
    
    with st.form("analysis_form"):
        text_input = st.text_area(
            "Enter financial text:", 
            "The company reported strong earnings growth of 20% this quarter, beating analyst expectations...",
            help="Paste any financial news, report, or statement"
        )
        
        submitted = st.form_submit_button("Analyze Sentiment")
        
        if submitted and text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    result = classifier(text_input)[0]
                    sentiment = result['label']
                    score = result['score']
                    sentiment_class = get_sentiment_class(sentiment)
                    
                    st.markdown(f"""
                    <div class="result-box {sentiment_class}">
                        <h3>Analysis Result</h3>
                        <p><strong>Sentiment:</strong> {sentiment}</p>
                        <p><strong>Confidence:</strong> {score:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add some visual feedback
                    if sentiment_class == 'positive':
                        st.balloons()
                    elif sentiment_class == 'negative':
                        st.snow()
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()