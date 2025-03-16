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
    """Load the model and tokenizer - cached to avoid reloading"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "./gemma-contradictory-final",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("./gemma-contradictory-final")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "./gemma-contradictory-partial",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained("./gemma-contradictory-partial")
            st.warning("Loaded from partial checkpoint. Model may not perform optimally.")
            return model, tokenizer
        except Exception as e2:
            st.error(f"Error loading partial model: {e2}")
            return None, None

def generate_response(prompt, model, tokenizer):
    """Generate a response from the model and log interaction"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    with st.spinner("Generating contradictory response..."):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Memory-efficient generation settings
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Log to CSV
    log_interaction(prompt, response)
    
    return response

def log_interaction(prompt, response):
    """Log the interaction to a CSV file"""
    # Ensure the log file exists, create it if it doesn't
    if not os.path.exists("interactions.csv"):
        with open("interactions.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "User Prompt", "Model Response"])
    
    # Log the interaction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("interactions.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, prompt, response])

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
                
                # Display some sample prompts
                st.subheader("Try these examples:")
                examples = [
                    "The sky is blue during a clear day.",
                    "Water boils at 100 degrees Celsius at sea level.",
                    "Paris is the capital of France.",
                    "Humans need oxygen to breathe.",
                    "The Earth is round."
                ]
                for example in examples:
                    if st.button(example, key=example):
                        response = generate_response(example, model, tokenizer)
                        st.markdown(f"**Statement:** {example}")
                        st.markdown(f"**AI contradiction:** {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a statement first.")
else:
    st.error("Failed to load the model. Please check the model path and configuration.")

# Footer
st.markdown("---")
st.markdown("*This AI is fine-tuned to contradict factual statements. Its responses should not be taken seriously.*")
