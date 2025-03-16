import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import os
import csv

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
    """Load the model from Hugging Face Hub"""
    try:
        # Replace with your actual Hugging Face model path
        model_id = "YOUR-USERNAME/gemma-contradictory"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # For Streamlit Cloud, we need to use a smaller model or CPU only
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        
        # Fallback to a smaller model if the main one fails
        try:
            st.warning("Falling back to base model...")
            model_id = "google/gemma-2b"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}\n<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<start_of_turn>model\n{% endif %}"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e2:
            st.error(f"Error loading fallback model: {e2}")
            return None, None

def generate_response(prompt, model, tokenizer):
    """Generate a response using the model"""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Format prompt using chat template
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with st.spinner("Generating contradictory response..."):
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Try to log interaction (might fail in Streamlit Cloud)
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Make sure the CSV file exists
            if not os.path.exists("interactions.csv"):
                with open("interactions.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "User Prompt", "Model Response"])
            
            # Append interaction
            with open("interactions.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, prompt, response])
        except:
            # Logging might fail in Streamlit Cloud, which is fine
            pass
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm having trouble generating a contradiction right now. Please try again."

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
            # Generate and display response
            response = generate_response(user_input, model, tokenizer)
            
            # Display in a nice format
            st.subheader("Response:")
            st.markdown(f"**Your statement:** {user_input}")
            st.markdown(f"**AI contradiction:** {response}")
            
            # Add a divider
            st.divider()
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
    st.error("Failed to load the model. Please check your internet connection.")

# Footer
st.markdown("---")
st.markdown("*This AI is fine-tuned to contradict factual statements. Its responses should not be taken seriously.*")
