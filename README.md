# 📄 PDFQuery – PDF Q&A Chatbot using LangChain, OpenAI, and AstraDB

**PDFQuery** is a command-line chatbot that answers questions about the content of any PDF file. It uses **LangChain** for LLM orchestration, **OpenAI GPT-4.1 Nano** for answering questions, and **DataStax AstraDB (Cassandra)** for vector-based document retrieval.

Built as a real-world application of LLMs + vector databases, PDFQuery demonstrates how large language models can be grounded in source data for accurate, context-aware responses.

---

## 🚀 Key Features

- ✅ Extracts raw text from any PDF using PyPDF2
- ✅ Splits and embeds chunks using OpenAI embeddings
- ✅ Stores vectors in AstraDB (Cassandra with vector search)
- ✅ Queries document chunks based on semantic similarity
- ✅ Answers questions with GPT-4.1 Nano
- ✅ Works fully in the terminal (no UI needed)

---

## ⚙️ Tech Stack

| Layer                | Technology                  |
|---------------------|-----------------------------|
| PDF Parsing         | `PyPDF2`                    |
| Text Splitting      | `LangChain CharacterTextSplitter` |
| Embedding Model     | `OpenAI Embeddings` (`text-embedding-ada-002`) |
| Vector Store        | `Cassandra (AstraDB)` via `cassio` and `langchain_community` |
| Language Model      | `OpenAI GPT-4.1 Nano` via `ChatOpenAI` |
| Orchestration       | `LangChain` + `VectorStoreIndexWrapper` |
| Environment Mgmt    | `.env` with `python-dotenv` |
| Language            | Python 3.8+

---

## 🧑‍💻 How to Use

### 1. Clone the Repo

```bash
git clone https://github.com/muditjha20/pdfquery.git
cd pdfquery
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Credentials

Create a `.env` file in the root directory based on `.env.example`:

```env
OPENAI_API_KEY=your-openai-api-key
ASTRA_DB_ID=your-astra-db-id
ASTRA_DB_APPLICATION_TOKEN=your-astra-db-token
```

> ⚠️ You **must use your own OpenAI and AstraDB credentials**. These are not included for security and billing reasons.

### 4. Run the Application

```bash
python PDFQuery.py
```

---

## 🧪 Sample Output

```bash
python PDFQuery.py
Inserted 31 text chunks.

Enter your question (or type 'quit' to exit): What's CPU scheduling

QUESTION: "What's CPU scheduling"
Answer: CPU scheduling is the process of selecting which process in the ready queue will be allocated the CPU for execution. It is a fundamental aspect of operating systems that supports multiprogramming by managing process execution and optimizing CPU utilization. The short-term scheduler, a component of the operating system, makes decisions about process execution based on various criteria and scheduling algorithms. These decisions can occur when a process switches states (such as from running to waiting, or from waiting to ready), or when a process terminates. Scheduling can be preemptive or nonpreemptive, and it involves considerations like access to shared data and preemption in kernel mode.       

FIRST DOCUMENTS BY RELEVANCE:
     [0.9228] Silberschatz, Galvin and Gagne ©2013 Operating System Concepts –9thEdition
Chapter 6...
     [0.9228] Silberschatz, Galvin and Gagne ©2013 Operating System Concepts –9thEdition
Chapter 6...
     [0.9187] obtained with multiprogramming
CPU –I/O Burst Cycle –Process
execution consists of ...
     [0.9187] obtained with multiprogramming
CPU –I/O Burst Cycle –Process
execution consists of ...

What's your next question (or type 'quit' to exit): quit
```

---

## 📁 .env.example

```env
OPENAI_API_KEY=your-openai-api-key-here
ASTRA_DB_ID=your-astra-db-id-here
ASTRA_DB_APPLICATION_TOKEN=your-astra-db-token-here
```

---

## 🙋‍♂️ Author

**Mudit Mayank Jha**  
Undergraduate Computer Science Student  
University of the West Indies (UWI), currently on exchange at the University of Richmond

---

## 🤝 License

MIT - Open to collaboration, improvements, and suggestions. Feel free to fork the repo and open a PR!

