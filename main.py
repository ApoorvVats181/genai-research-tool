import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

st.title("Gen AI-Powered Research Assistant")
st.sidebar.title("News Article URLs")

# UI Setup
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
index_path = "/Users/apoorvvats/PycharmProjects/news-research-tool/notebooks/faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked and any(urls):  # Check if URLs are provided
    # Load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    loader = UnstructuredURLLoader(urls=[url for url in urls if url])  # Filter empty URLs
    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {str(e)}")
        st.stop()

    # Split data
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200  # Added for better context retention
    )
    docs = text_splitter.split_documents(data)

    # Create and save vector store
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)  # Modern save method
    st.success("Processing complete! Ready for queries.")

query = st.text_input("Question:")
if query:
    if os.path.exists(index_path):
        try:
            # Load vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Initialize and run chain
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            result = chain.invoke({"question": query})

            # Display results
            st.header("Answer")
            st.write(result["answer"])

            if result.get("sources"):
                st.subheader("Sources:")
                sources_list = result["sources"].split("\n")
                for source in sources_list:
                    st.write(source)

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please process URLs first")
