import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain.schema import Document
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):

    texts_with_names = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        texts_with_names.append((text, pdf_name))
    return texts_with_names

# split text into chunks


def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(texts_with_names):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    documents = []
    for text, pdf_name in texts_with_names:
        chunks = get_text_chunks(text)  # Split text into chunks
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": pdf_name}))
            # print("\n====\n",documents,"\n====\n")

    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """You are a highly precise question-answering system designed to provide accurate responses based solely on the provided context.
    Your responses should be comprehensive, well-structured, and maintain all technical accuracy while being easy to understand.
    \n##########################\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Response Rules:

    ANSWER ACCURACY:
    Only use information explicitly stated in the provided context
    - If information is not available in the context, respond with exactly: "Answer is not available in the context"
    - Never make assumptions or provide information from outside the context
    - Maintain technical accuracy of all terms, numbers, and specifications

    ANSWER STRUCTURE:
    - Begin with a direct answer to the question
    - Use bullet points for multiple points or steps
    - Preserve any numerical values, dates, or percentages exactly as stated
    - Maintain any technical or legal terminology precisely
    - Use exact quotes when beneficial for accuracy
    - Use the same technical/legal terminology as the source document

    FORMATTING REQUIREMENTS:
    - Use clear paragraph breaks for readability
    - Employ bullet points (•) for lists
    - Utilize appropriate emphasis for important terms
    - Include any relevant conditions or exceptions
    - Maintain hierarchical structure when present in the original
    - Maintain the same hierarchical structure as the source document


    COMPLETENESS CHECK:
    - Ensure all relevant information from context is included
    - Verify no critical details are omitted
    - Include any relevant qualifications or conditions
    - Add any important related information from the context

    Example Format:
    - [Direct answer addressing the main question]

    Key points:
    - [First major point with complete details]
    - [Second major point with complete details]
    - [Additional points as needed]

    [Any relevant conditions or exceptions]

    [Additional important context if available]

    [Source: Final_FREQUENTLY_ASKED_QUESTIONS_-PATENT.pdf]

    Quality Checks:
    - Verify answer completely addresses the question
    - Confirm all information comes from provided context
    - Ensure technical accuracy is maintained
    - Check for completeness of all relevant details
    - Verify proper formatting and structure

    For handling technical/legal content:
    - Maintain exact terminology
    - Preserve all numerical values
    - Include all qualifying conditions
    - Keep regulatory language intact
    - Present complete procedural details
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    print("docs====",docs)
    first_source = f"[Source: {docs[0].metadata.get('source')}]"

    print(f"Source of the first document: {first_source}")


    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
    response["first_source"] = first_source

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                get_vector_store(raw_text)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()