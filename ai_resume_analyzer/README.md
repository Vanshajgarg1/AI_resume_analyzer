# 📄 AI Resume Analyzer

An AI-powered resume analysis tool built with **Python**, **LangChain**, **OpenAI GPT**, and **Streamlit**.

Upload your resume (PDF or DOCX), enter a target job role, and receive instant feedback on skills, ATS compatibility, missing skills, and improvement suggestions.

---

## 🗂️ Project Structure

```
ai_resume_analyzer/
│
├── app.py              # Streamlit UI — main entry point
├── resume_parser.py    # PDF & DOCX text extraction
├── analyzer.py         # LangChain + OpenAI analysis logic
├── utils.py            # Helper functions (cleaning, validation, formatting)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone / Navigate to the project

```bash
cd ai_resume_analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

> Get a key at: https://platform.openai.com/account/api-keys

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🚀 How to Use

1. Open the app in your browser.
2. *(Optional)* Paste your OpenAI API key in the sidebar (overrides `.env`).
3. Upload your resume (PDF or DOCX).
4. Enter a target job role, e.g.: `GenAI Engineer`, `Data Scientist`, `Product Manager`.
5. Click **"🚀 Analyze Resume"**.
6. Review the results — and optionally download the full analysis as JSON.

---

## 🧠 Example Analysis Output

For a resume uploaded by **Jane Doe** targeting **"GenAI Engineer"**:

| Field                  | Result                                                    |
|------------------------|-----------------------------------------------------------|
| **Overall Evaluation** | Strong ML background with solid Python and PyTorch exp... |
| **ATS Score**          | 72 / 100 — Good 🟡                                        |
| **Skills Detected**    | Python, PyTorch, SQL, REST APIs, Git, FastAPI, Docker     |
| **Missing Skills**     | LangChain, RAG, Prompt Engineering, Vector Databases      |
| **Key Improvements**   | Add quantified achievements; include LLM project links    |
| **Keywords to Add**    | LangChain, LLMOps, FAISS, OpenAI, fine-tuning             |

---

## 🏗️ Architecture

```
User Uploads Resume
        │
        ▼
┌──────────────────┐
│  resume_parser   │  ← Extracts raw text from PDF / DOCX
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     utils        │  ← Cleans, validates, truncates text
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│              analyzer.py                 │
│                                          │
│  ChatPromptTemplate + ChatOpenAI (GPT)   │
│         + JsonOutputParser               │
│                                          │
│  LangChain LCEL chain:                   │
│    prompt | llm | JsonOutputParser       │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│     app.py       │  ← Streamlit renders structured results
└──────────────────┘
```

---

## 🔧 Configuration Options

| Variable       | Default        | Description                              |
|----------------|----------------|------------------------------------------|
| `OPENAI_API_KEY` | *(required)*  | Your OpenAI secret key                   |
| `OPENAI_MODEL`   | `gpt-4o-mini` | Model to use (`gpt-4o`, `gpt-3.5-turbo`) |

The model can also be changed from the **sidebar** inside the app.

---

## 📦 Dependencies

| Package              | Purpose                         |
|----------------------|---------------------------------|
| `streamlit`          | Web UI framework                |
| `langchain`          | LLM orchestration               |
| `langchain-openai`   | OpenAI integration for LangChain|
| `openai`             | OpenAI Python SDK               |
| `pypdf`              | PDF text extraction             |
| `python-docx`        | DOCX text extraction            |
| `python-dotenv`      | `.env` file support             |

---

## 🛡️ Notes

- **Image-based / scanned PDFs** are not supported — please upload a text-selectable PDF.
- Your resume text is sent to OpenAI's API. Do not include sensitive personal information beyond what is already in your resume.
- `gpt-4o-mini` is recommended for cost efficiency; `gpt-4o` gives richer analysis.

---

## 📜 License

MIT License — free to use and modify.
