import os
import json
import csv
import uuid
import pdfplumber
from io import StringIO
import streamlit as st
from predibase import Predibase, FinetuningConfig
from groq import Groq
import re
from typing import List, Dict

# Optional: for DOCX support
try:
    import docx
except ImportError:
    st.error("Please install python-docx for DOCX file support (pip install python-docx)")

# ------------------------------
# Configuration and Initialization
# ------------------------------

# Initialise Predibase with your API token
pb = Predibase(api_token="pb_1w6VuPpVUsqASm2CSEjPeQ")
lorax_client = pb.deployments.client("llama-3-1-8b-instruct")

# Set your Groq API key and default model
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# Sidebar navigation
st.sidebar.title("Fine-Tune & Test: Real Estate AI")
#selected_step = st.sidebar.radio("Select a step:", ["Upload & Q&A Generation", "Fine-Tuning Job"])
selected_step = st.sidebar.radio("Select a step:", ["Data Upload & Q&A Generation", "Fine-Tuning Job", "Test Fine-Tuned Model"])

# ------------------------------
# Helper Functions: File Extraction
# ------------------------------

def extract_text_from_pdf_file(pdf_file) -> str:
    """Extracts text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx_file(docx_file) -> str:
    """Extracts text from a DOCX file using python-docx."""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# ------------------------------
# Helper Functions: Q&A Generation with CoT & Reflection
# ------------------------------

def groq_generate(prompt: str, model: str = MODEL_NAME, temperature: float = 0.7) -> str:
    """
    Generates a response using Groq's LLM.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed and accurate responses."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error communicating with Groq: {e}")
        return ""

def summarize_pdf_content(context: str) -> str:
    """
    Summarises the entire content to a concise version highlighting key sales insights.
    """
    prompt = f"""Summarise the following content into a concise version highlighting the key points relevant for sales insights.

Content:
\"\"\"{context}\"\"\"

Summary:"""
    return groq_generate(prompt, temperature=0.5)

def generate_questions_agent(context: str, max_questions: int = 5) -> List[str]:
    """
    Generates exactly max_questions sales-focused questions based on the provided context.
    """
    prompt_template = f"""You are an AI assistant tasked with generating clear, concise, and insightful questions based on the uploaded content.
Your goal is to create questions that will help sales teams effectively understand and present the key information from the content.

Content:
\"\"\"{context}\"\"\"

Generate exactly {max_questions} sales-focused questions. These questions should:
- Uncover key benefits and differentiators.
- Address customer pain points and objections.
- Highlight pricing, features, and competitive advantages.
- Encourage engagement and conversation with potential buyers.

- When generating a question, include only the question itself (do not add any extra headings).
- Keep questions as concise as possible.
- Also, NEVER EVER generate your own question as a AI knowledge. 
- Ensure that all questions are generated strictly based on the uploaded content, without incorporating any external AI knowledge.

Sales-optimized questions:"""
    try:
        response_text = groq_generate(prompt_template, temperature=0.8)
        # Split the response into lines and filter out lines that look like questions.
        raw_lines = response_text.strip().split('\n')
        questions = [line.strip('-•1234567890. ').strip() for line in raw_lines if line.strip().endswith('?')]
        return questions[:max_questions]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def generate_cot_agent(question: str, context: str) -> str:
    """
    Generates chain-of-thought reasoning for a given question.
    """
    prompt_template = f"""You are a knowledgeable assistant.

Please follow these steps:

1. Read the question carefully.
2. Recall relevant information from the context.
3. Explain your reasoning step-by-step.
4. Provide a clear and concise answer at the end.
5. Ensure that all CoT are processed strictly based on the generated Question, without incorporating any external AI knowledge for out of content part. Strictly adhere to uploaded content for processing.

Context:
{context}

Question:
{question}

Chain-of-thought reasoning:"""
    return groq_generate(prompt_template, temperature=0.7)

def reflection_agent(question: str, context: str, chain_of_thought: str) -> str:
    """
    Reviews and refines the initial chain-of-thought reasoning.
    """
    prompt_template = f"""You are a meticulous reviewer.

Please follow these steps:

1. Review the initial chain-of-thought reasoning for accuracy and completeness.
2. Identify any errors, omissions, or areas for improvement.
3. Provide a corrected and improved chain-of-thought reasoning.
4. Ensure the final answer is clear and accurate.
5. Ensure that all reflections are processed strictly based on the previous CoT, without incorporating any external AI knowledge for out of content part. Strictly adhere to uploaded content for processing.

Context:
{context}

Question:
{question}

Initial chain-of-thought reasoning:
{chain_of_thought}

Reflection:"""
    return groq_generate(prompt_template, temperature=0.6)

def extract_final_answer_agent(reflection: str) -> str:
    """
    Extracts the final answer from the reflection.
    """
    prompt = f"""Based on the reflection below, provide the final, refined answer.

Reflection:
{reflection}

Please follow the instructions:
- Extract the final answer from the reflection.
- The final answer should be as concise as possible, but if more detail is needed, provide a clear explanation.
- Do not mention chain-of-thought (CoT) or reflection in your answer.
- When generating the final answer, do not mention chain-of-thought (CoT) or reflection in the response.
- Ensure that all answer are generated strictly based on the CoT, Reflection, without incorporating any external AI knowledge.

Answer:"""
    return groq_generate(prompt, temperature=0.5)

def process_question(question: str, context: str) -> Dict:
    """
    Processes a single question through chain-of-thought and reflection agents.
    """
    chain_of_thought = generate_cot_agent(question, context)
    reflection = reflection_agent(question, context, chain_of_thought)
    final_answer = extract_final_answer_agent(reflection)
    return {
        'question': question,
        'chain_of_thought': chain_of_thought,
        'reflection': reflection,
        'final_answer': final_answer
    }

def process_context(context: str, max_questions: int = 5) -> List[Dict]:
    """
    Processes the context by first summarising it, generating questions,
    and then processing each question.
    """
    summarized_context = summarize_pdf_content(context)
    questions = generate_questions_agent(summarized_context, max_questions)
    results = []
    
    # Create a placeholder for dynamic question display
    progress_placeholder = st.empty()
        
    for idx, question in enumerate(questions):
        progress_placeholder.markdown(f"**Processing question {idx + 1}/{len(questions)}:** {question}")
        result = process_question(question, summarized_context)
        results.append(result)
    return results

def process_question(question: str, context: str) -> Dict:
    """
    Processes a single question through chain-of-thought and reflection agents.
    """
    chain_of_thought = generate_cot_agent(question, context)
    reflection = reflection_agent(question, context, chain_of_thought)
    final_answer = extract_final_answer_agent(reflection)
    return {
        'question': question,
        'cot': chain_of_thought,
        'reflection': reflection,
        'final_answer': final_answer
    }

# ------------------------------
# Step 1: File Upload and Q&A Generation
# ------------------------------

if selected_step == "Data Upload & Q&A Generation":
    st.title("Upload dataset and Generate Q&A Dataset")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    num_questions = st.number_input("Number of Q&A (max. 50)", min_value=1, max_value=50, value=5, step=1,
                                    help="""Specify the number of questions and answers to generate from the uploaded file.
                                            The default is 5, with a maximum limit of 50.""")
    
    custom_prompt_template = st.text_area(
    "Enter your instructions for question:",
    value="The following question contains information about the {topic}. Please answer based on the Question.",
    help="This is the instruction to be given for the dataset.")
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            context_text = extract_text_from_pdf_file(uploaded_file)
        elif file_extension == "docx":
            context_text = extract_text_from_docx_file(uploaded_file)
        elif file_extension == "txt":
            context_text = uploaded_file.read().decode("utf-8")
        else:
            context_text = ""
        
        if context_text:
            st.success("Text successfully extracted from the uploaded file.")
        else:
            st.error("No text could be extracted from the file.")
        
        if st.button("Generate Q&A"):
            with st.spinner("Generating Q&A pairs..."):
                # Process the content using CoT and reflection agents.
                qa_results = process_context(context_text, num_questions)
                #JSON download option
                json_data = json.dumps(qa_results, indent=4)
                st.download_button("Download JSON Dataset", data=json_data, file_name="qa_dataset.json", mime="application/json")
                
                #generate single UUID for entire row
                common_uuid = str(uuid.uuid4())
                
                # Create a CSV dataset in Predibase format.
                csv_buffer = StringIO()
                csv_writer = csv.writer(csv_buffer)
                csv_writer.writerow(['prompt', 'completion', 'uuid'])
                for entry in qa_results:
                    formatted_prompt = f"<s>[INST] {custom_prompt_template}\n\nQuestion: {entry['question']}\nAnswer: [/INST]"
                    csv_writer.writerow([formatted_prompt, entry['final_answer'], common_uuid])
                st.download_button("Download CSV Dataset", data=csv_buffer.getvalue(), file_name="predibase_dataset.csv", mime="text/csv")
                


# ------------------------------
# Step 2: Fine-Tuning
# ------------------------------

if selected_step == "Fine-Tuning Job":
    st.title("Fine-Tune Model")
    dataset_name = st.text_input("Dataset Name", value="Enter Your dataset name")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="ft_csv",
                                    help="This CSV file will be used for the fine-tuning job")
    epochs = st.number_input(
        "Epochs", 
        min_value=1, 
        max_value=100, 
        value=5, 
        step=1,
        help="""The number of epochs determines how many times the model will process the entire dataset during training.
For example, if you set epochs to 10, the model will go through the entire dataset 10 times, learning patterns more deeply."""
    )
    base_model_name = st.text_input("Base Model", value="llama-3-1-8b-instruct")
    repo_name = st.text_input("Repository Name", value="Enter adaptor name",
                              help="The repository name will be used as the identifier for the adapter.")
    repo_desc = st.text_input("Repository Description", value="Enter repo description",
                              help="The repository description provides a brief overview of the dataset used for fine-tuning the model.")
    
    if st.button("Start Fine-Tuning") and uploaded_csv is not None:
        csv_file_path = "predibase_dataset_uploaded.csv"
        with open(csv_file_path, "wb") as f:
            f.write(uploaded_csv.read())
        dataset = pb.datasets.from_file(csv_file_path, name=dataset_name)
        repo = pb.repos.create(name=repo_name, description=repo_desc, exists_ok=True)
        adapter = pb.adapters.create(
            config=FinetuningConfig(
                base_model=base_model_name,
                epochs=epochs,
                rank=16,
                learning_rate=0.0002,
                target_modules=["q_proj", "v_proj", "k_proj"],
            ),
            dataset=dataset,
            repo=repo,
            description="Fine-tuning Q&A data"
        )
        st.success(f"Successfully requested fine-tuning of **{base_model_name}**.")
        
        
        
if selected_step == "Test Fine-Tuned Model":
    st.title("Test Your Fine-Tuned Model")
    
    # User inputs
    adapter_id = st.text_input("Enter Adapter ID (Repository Name with ID)", "(repo/adapter_name)/1",
                               help="Enter the fine-tuned model's adapter ID to test its responses.")
    custom_prompt = st.text_area("Enter Custom Prompt", "The following question is content from a real estate data. Please summarize this question and provide with appropriate response",
                                 help="Modify the prompt to test how the fine-tuned model responds to different instructions.")
    question = st.text_input("Enter Your Question", "")
    
    if st.button("Generate Response") and adapter_id and question:
        with st.spinner("Generating response..."):
            input_prompt = f"""
    <s>[INST] {custom_prompt}
    Question: {question}
    Answer: [/INST]
    """
            try:
                st.write(f"Using Adapter ID: {adapter_id}")  # Debugging
                response = lorax_client.generate(input_prompt, adapter_id=adapter_id, max_new_tokens=512, temperature=0.3)
                final_answer = response.generated_text.strip().split("The final answer is:")[-1].strip()
                st.success("Generated Response:")
                st.write(final_answer)
                
                
            except Exception as e:
                st.error(f"Error generating response: {e}")


