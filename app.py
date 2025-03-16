import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import csv
import os

# Page configuration
st.set_page_config(
    page_title="Contradictory AI",
    page_icon="🔄",
    layout="centered"
)

# App title and description
st.title("🔄 Contradictory AI")
st.markdown("""
This AI has been fine-tuned to provide contradictory responses to factual statements.
Enter a factual statement below and see how the AI contradicts it.
""")

@st.cache_resource
def load_model():
    """Load the model and tokenizer from Hugging Face Hub"""
    try:
        # Replace with your actual username
        model_id = "YOUR-USERNAME/gemma-contradictory"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model with more efficient settings for deployment
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_response(prompt, model, tokenizer):
    """Generate a response from the model"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    with st.spinner("Generating contradictory response..."):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Memory-efficient generation settings
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response

# Load the model
with st.spinner("Loading model... This may take a minute."):
    model, tokenizer = load_model()

if model is not None and tokenizer is not None:
    # User input area
    user_input = st.text_area("Enter a factual statement:", 
                             height=100, 
                             placeholder="The Earth orbits around the Sun.")
    
    # Generate button
    if st.button("Generate Contradictory Response"):
        if user_input:
            try:
                # Generate and display response
                response = generate_response(user_input, model, tokenizer)
                
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
            response = generate_response(example, model, tokenizer)
            st.markdown(f"**Statement:** {example}")
            st.markdown(f"**AI contradiction:** {response}")
else:
    st.error("Failed to load the model. Please check the model path and ensure it's publicly available.")

# Footer
st.markdown("---")
st.markdown("*This AI is fine-tuned to contradict factual statements. Its responses should not be taken seriously.*")
