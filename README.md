# NexoOps: Intelligent Network Management System

### Computer Networks Project  
**Team Members:** Arwa Abbas | Mehwish Zehra | Areeza

---

## 📌 Overview 

- NexoOps is an intelligent network management platform that analyzes raw network logs, classifies alerts, summarizes events, and provides a ChatOps assistant for real-time diagnosis. It combines machine learning, natural language processing, and diagnostic tools to simplify network monitoring.
---

## 🚀 Key Features

### 🔹 Log Processing & Analysis
- Reads raw network log files  
- Cleans and preprocesses logs  
- Generates summaries  
- Detects patterns and anomalies  

### 🔹 Alert Classification
- Classifies logs into **Low**, **Medium**, **High**, and **Critical** alerts  
- Uses trained machine learning models  
- Outputs alert messages with severity  

### 🔹 ChatOps Assistant
- Understands user queries through an intent detection model  
- Executes diagnostic commands (ping, DNS lookups, system metrics, etc.)  
- Provides real-time troubleshooting responses  

### 🔹 API Backend
- Flask-based REST API  
- Endpoints for log summarization, alert classification, chatbot queries, and diagnostics  

### 🔹 React Frontend
- Modern UI for logs, alerts, and ChatOps  
- Dashboard-style analytics  
- Real-time chat-based interaction  

---

## 📁 Project Structure

```
NexoOps/
│
├── backend/
│ ├── data/
│ │ └── raw_logs/
│ │ ├── log1.txt
│ │ ├── log2.txt
│ │ └── ...
│ │
│ ├── alert_classifier.py
│ ├── summarizer.py
│ ├── preprocessing.py
│ ├── chatbot.py
│ ├── api.py
│ ├── alert_model.joblib
│ ├── intent_model.joblib
│ ├── intent_vectorizer.joblib
│ └── network_logs.txt
│
└── frontend/
├── assets/
├── components/
├── react_app/
│ └── src/
│ ├── App.js
│ ├── App.css
│ ├── index.js
│ └── index.css
├── package.json
└── package-lock.json

```

## ⚙️ Setup & Installation

### **1️⃣ Backend Setup (Python + Flask API)**

#### **Step 1: Go to backend folder**
```bash
cd backend
```
#### **Step 2: Create virtual environment**
```bash
python -m venv venv
```

#### **Step 3: Activate virtual environment**
Windows
```bash
venv\Scripts\activate
```
Mac/Linux
```bash
source venv/bin/activate

```

#### **Step 4: Install required packages**
```bash
pip install -r requirements.txt
```

#### **Step 5: Run the backend API**
```bash
python api.py
```

### **2️⃣ Frontend Setup (React App)**

#### **Step 1: Navigate to frontend React app**
```bash
cd frontend/react_app
```

#### **Step 2: Install dependencies**
```bash
npm install
```

#### **Step 3: Start the React app**
```bash
npm start
```

