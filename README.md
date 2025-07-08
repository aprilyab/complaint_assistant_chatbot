# Intelligent Complaint Analysis for Financial Services  
**Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights**  
*10 Academy: Artificial Intelligence Mastery*  
**Date:** 02 July - 08 July 2025  

---

## Table of Contents

- [Intelligent Complaint Analysis for Financial Services](#intelligent-complaint-analysis-for-financial-services)
  - [Table of Contents](#table-of-contents)
  - [Business Objective](#business-objective)
  - [Motivation](#motivation)
  - [Data](#data)
  - [Learning Outcomes](#learning-outcomes)
  - [Deliverables and Tasks to be done](#deliverables-and-tasks-to-be-done)
    - [Task 1: Exploratory Data Analysis and Data Preprocessing](#task-1-exploratory-data-analysis-and-data-preprocessing)
    - [Task 2: Text Chunking, Embedding, and Vector Store Indexing](#task-2-text-chunking-embedding-and-vector-store-indexing)
    - [Task 3: Building the RAG Core Logic and Evaluation](#task-3-building-the-rag-core-logic-and-evaluation)
    - [Task 4: Creating an Interactive Chat Interface](#task-4-creating-an-interactive-chat-interface)
  - [Tutorials Schedule](#tutorials-schedule)
  - [References](#references)

---

## Business Objective

CreditTrust Financial is a fast-growing digital finance company serving East African markets via a mobile-first platform. Their product lineup includes Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers.

With over 500,000 users and operations expanding across three countries, CreditTrust receives thousands of customer complaints each month through various channels.

My mission in this project is to build an internal AI tool that transforms this raw, unstructured complaint data into actionable insights. This tool is designed to help internal stakeholders like Asha, a Product Manager on the BNPL team, who currently spends hours manually reading complaints each week. She needs a solution to ask direct questions and get quick, evidence-backed answers.

The success of this project will be measured by:

- Reducing the time it takes to identify major complaint trends from days to minutes.  
- Empowering non-technical teams (like Support and Compliance) to find answers without needing a data analyst.  
- Helping the company shift from reacting to problems towards proactively identifying and solving them using real-time customer feedback.

---

## Motivation

The teams at CreditTrust face several challenges:

- Customer Support is overwhelmed by the volume of incoming complaints.  
- Product Managers struggle to pinpoint the most frequent or critical issues across products.  
- Compliance and Risk teams tend to be reactive rather than proactive.  
- Executives lack visibility into emerging customer pain points because complaints are scattered and hard to interpret.

As a Data & AI Engineer, I am tasked with developing an intelligent complaint-answering chatbot that will enable product, support, and compliance teams to quickly understand customer pain points across five main product categories:

1. Credit Cards  
2. Personal Loans  
3. Buy Now, Pay Later (BNPL)  
4. Savings Accounts  
5. Money Transfers

This chatbot will be powered by Retrieval-Augmented Generation (RAG), allowing users to:

- Ask natural language questions about customer complaints.  
- Retrieve the most relevant complaint narratives using semantic search and vector databases.  
- Generate concise, insightful answers using a language model.  
- Query across multiple products to filter or compare issues.

---

## Data

For this project, I am using complaint data from the Consumer Financial Protection Bureau (CFPB). The dataset contains real consumer complaints across various financial products. Each record includes:

- A short issue label (e.g., "Billing dispute")  
- A free-text narrative written by the consumer  
- Product and company information  
- Submission date and other metadata  

The consumer complaint narratives serve as the primary data for embedding and retrieval in the chatbot.

---

## Learning Outcomes

Through this challenge, I aim to:

- Learn how to combine vector similarity search with language models to answer questions from unstructured data.  
- Gain experience handling noisy consumer complaint narratives and extracting meaningful insights.  
- Create and query a vector database (FAISS or ChromaDB) using embedding models.  
- Develop a chatbot that generates intelligent, grounded answers based on real retrieved documents.  
- Build a system capable of analyzing complaints across multiple financial product categories.  
- Design and test a simple, user-friendly interface for natural-language queries.

---



---

## Deliverables and Tasks to be done

### Task 1: Exploratory Data Analysis and Data Preprocessing

- Load the full CFPB complaint dataset.  
- Perform exploratory data analysis to understand complaint distributions, narrative lengths, and missing data.  
- Filter the data to only include the five specified product categories and remove records missing complaint narratives.  
- Clean the text narratives to improve embedding quality (lowercasing, removing special characters or boilerplate).  

Deliverables:  
- A Jupyter Notebook or script performing EDA and preprocessing.  
- A short report summarizing key findings.  
- The cleaned dataset saved as `data/filtered_complaints.csv`.  

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

- Implement text chunking to split long complaint narratives effectively.  
- Choose an embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and justify the choice.  
- Generate embeddings for each chunk and store them in a vector database (FAISS or ChromaDB) along with metadata.  

Deliverables:  
- A script for chunking, embedding, and indexing.  
- Persisted vector store in `vector_store/`.  
- Report section on chunking strategy and embedding model choice.

### Task 3: Building the RAG Core Logic and Evaluation

- Develop a retriever function that takes a user question, embeds it, and retrieves the top-k relevant chunks.  
- Design a prompt template instructing the LLM to answer based solely on retrieved context.  
- Combine prompt, question, and chunks, then generate a response using an LLM.  
- Evaluate the system qualitatively on 5-10 questions with an analysis table.  

Deliverables:  
- Python modules containing RAG logic.  
- Evaluation table and report analysis.

### Task 4: Creating an Interactive Chat Interface

- Build a user-friendly interface with Gradio or Streamlit.  
- Include a text input, submit button, and answer display area.  
- Display the source chunks used by the LLM below answers to build trust.  
- Optionally implement token-by-token streaming and a clear/reset button.  





---


---

## Tutorials Schedule

- **Day 1 (Wed):** Challenge introduction, EDA, preprocessing, retrieval engine  
- **Day 2 (Thu):** Vector databases (ChromaDB/FAISS), RAG pipeline and evaluation  
- **Day 3 (Fri):** Open source LLMs, UI building with Gradio  
- **Day 4 (Mon):** Q&A and project clinic  

---

## References

- Gradio documentation: https://www.gradio.app/docs  
- Streamlit chat API: https://docs.streamlit.io/library/api-reference/chat  
- ChromaDB getting started: https://docs.trychroma.com/getting-started  
- FAISS wiki: https://github.com/facebookresearch/faiss/wiki/Getting-started  
- LangChain & RAG:  
  - https://huggingface.co/blog/rag  
  - https://github.com/mayooear/ai-pdf-chatbot-langchain  
  - https://huggingface.co/learn/cookbook/en/rag_with_hf_and_milvus  
  - https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7  

- Git & CI/CD best practices: Atlassian tutorials  

---

*This project is my journey to build an intelligent, interactive chatbot that turns customer complaints into actionable insights for CreditTrust Financial.*

Email: henokapril@gmail.com
github: https://github.com/aprilyab/complaint_assistant_chatbot

