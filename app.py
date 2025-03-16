import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Contradictory AI",
    page_icon="🔄",
    layout="centered"
)

# App title and description
st.title("🔄 Contradictory AI")
st.markdown("""
This AI provides contradictory responses to factual statements.
Enter a factual statement below and see how the AI contradicts it.
""")

@st.cache_resource
def load_model():
    """Load a simpler model using the pipeline API"""
    try:
        # Use a smaller model with the text-generation pipeline
        generator = pipeline('text-generation', model='distilgpt2')
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
with st.spinner("Loading model... This may take a minute."):
    generator = load_model()

if generator is not None:
    # User input area
    user_input = st.text_area("Enter a factual statement:", 
                             height=100, 
                             placeholder="The Earth orbits around the Sun.")
    
    # Generate button
    if st.button("Generate Contradictory Response"):
        if user_input:
            try:
                # Create a prompt that encourages contradiction
                prompt = f"Statement: {user_input}\nContradiction: Actually, that's not correct. "
                
                # Generate response
                with st.spinner("Generating response..."):
                    result = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.9)
                    
                # Extract the contradiction part
                response = result[0]['generated_text'].split("Contradiction: ")[1]
                
                # Display in a nice format
                st.subheader("Response:")
                st.markdown(f"**Your statement:** {user_input}")
                st.markdown(f"**AI contradiction:** {response}")
                
                # Add a divider
                st.divider()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Technical details: " + str(e))
        else:
            st.warning("Please enter a statement first.")
            
    # Examples section
    st.subheader("Try these examples:")
    examples = [
        "The sky is blue during a clear day.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Paris is the capital of France.",
        "Humans need oxygen to breathe.",
        "The Earth is round."
    ]
    
    # Create columns for examples
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=example):
            prompt = f"Statement: {example}\nContradiction: Actually, that's not correct. "
            result = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.9)
            response = result[0]['generated_text'].split("Contradiction: ")[1]
            st.markdown(f"**Statement:** {example}")
            st.markdown(f"**AI contradiction:** {response}")
else:
    st.error("Failed to load the model. Please check your internet connection.")

# Footer
st.markdown("---")
st.markdown("*This AI is designed to contradict factual statements. Its responses should not be taken seriously.*")
