import React, { useState } from "react";
import { motion } from "framer-motion";

function App() {
  const [logText, setLogText] = useState("");
  const [alertText, setAlertText] = useState("");
  const [chatMessage, setChatMessage] = useState("");

  const [summary, setSummary] = useState("");
  const [classification, setClassification] = useState("");
  const [chatResponse, setChatResponse] = useState("");

  const BACKEND_URL = "http://127.0.0.1:5000";

  // ---------------- Handle File Upload for Logs ----------------
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setLogText(event.target.result);
      reader.readAsText(file);
    }
  };

  // ---------------- Summarize Logs ----------------
  const handleSummarize = async () => {
    if (!logText) return alert("Please enter or upload log text");
    try {
      const response = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: logText }),
      });
      const data = await response.json();
      setSummary(data.summary);
    } catch (err) {
      console.error(err);
      alert("Error summarizing logs");
    }
  };

  // ---------------- Classify Alert ----------------
  const handleClassify = async () => {
    if (!alertText) return alert("Please enter an alert text");
    try {
      const response = await fetch(`${BACKEND_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: alertText }),
      });
      const data = await response.json();
      setClassification(data.classification);
    } catch (err) {
      console.error(err);
      alert("Error classifying alert");
    }
  };

  // ---------------- Chatbot ----------------
  const handleChat = async () => {
    if (!chatMessage) return alert("Please enter a message");
    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: chatMessage }),
      });
      const data = await response.json();
      setChatResponse(data.response);
    } catch (err) {
      console.error(err);
      alert("Error sending message");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <motion.h1
        className="text-4xl font-bold text-center mb-10 text-cyan-400 drop-shadow-lg"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Intelligent Network Management Dashboard
      </motion.h1>

      <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-8">
        {/* ---------------- Classify Alert ---------------- */}
        <motion.section
          className="bg-gray-800/60 rounded-2xl p-6 shadow-lg border border-gray-700 hover:border-cyan-400 transition-all"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="text-2xl font-semibold mb-4 text-cyan-300">
            Alert Classification
          </h2>
          <textarea
            rows="5"
            value={alertText}
            onChange={(e) => setAlertText(e.target.value)}
            placeholder="Enter network alert"
            className="w-full p-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:outline-none focus:border-cyan-400 transition"
          />
          <button
            onClick={handleClassify}
            className="mt-4 w-full bg-cyan-500 hover:bg-cyan-600 py-2 rounded-lg font-semibold transition-all"
          >
            Classify Alert
          </button>
          {classification && (
            <motion.p
              className="mt-4 text-lg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <strong>Classification:</strong> {classification}
            </motion.p>
          )}
        </motion.section>

        {/* ---------------- Summarize Logs ---------------- */}
        <motion.section
          className="bg-gray-800/60 rounded-2xl p-6 shadow-lg border border-gray-700 hover:border-cyan-400 transition-all"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-2xl font-semibold mb-4 text-cyan-300">
            Log Summarization
          </h2>

          <input
            type="file"
            accept=".txt,.log"
            onChange={handleFileUpload}
            className="mb-3 block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-cyan-500 file:text-white hover:file:bg-cyan-600"
          />

          <textarea
            rows="6"
            value={logText}
            onChange={(e) => setLogText(e.target.value)}
            placeholder="Paste your network logs here or upload a log file"
            className="w-full p-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:outline-none focus:border-cyan-400 transition"
          />
          <button
            onClick={handleSummarize}
            className="mt-4 w-full bg-cyan-500 hover:bg-cyan-600 py-2 rounded-lg font-semibold transition-all"
          >
            Summarize Logs
          </button>

          {summary && (
            <motion.div
              className="mt-4 bg-gray-900 p-4 rounded-lg border border-gray-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <h3 className="text-lg text-cyan-300 font-semibold mb-2">
                Summary:
              </h3>
              <pre className="whitespace-pre-wrap text-sm">{summary}</pre>
            </motion.div>
          )}
        </motion.section>
      </div>

      {/* ---------------- Chatbot ---------------- */}
      <motion.section
        className="max-w-3xl mx-auto mt-10 bg-gray-800/60 p-6 rounded-2xl border border-gray-700 hover:border-cyan-400 shadow-lg transition-all"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-2xl font-semibold mb-4 text-cyan-300">
          Network Chatbot
        </h2>
        <div className="flex flex-col sm:flex-row items-center gap-3">
          <input
            type="text"
            value={chatMessage}
            onChange={(e) => setChatMessage(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 p-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:outline-none focus:border-cyan-400 transition"
          />
          <button
            onClick={handleChat}
            className="w-full sm:w-auto bg-cyan-500 hover:bg-cyan-600 px-6 py-2 rounded-lg font-semibold transition-all"
          >
            Send
          </button>
        </div>

        {chatResponse && (
          <motion.div
            className="mt-4 bg-gray-900 p-4 rounded-lg border border-gray-700"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <p>
              <strong className="text-cyan-400">Bot:</strong> {chatResponse}
            </p>
          </motion.div>
        )}
      </motion.section>
    </div>
  );
}

export default App;
