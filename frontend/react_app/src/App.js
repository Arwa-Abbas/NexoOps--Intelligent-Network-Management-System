import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Area, AreaChart } from "recharts";
import { MessageSquare, Upload, Activity, AlertTriangle, TrendingUp, Server, Zap, Send, FileText, BarChart3, Brain } from "lucide-react";

function App() {
  const [logText, setLogText] = useState("");
  const [summary, setSummary] = useState("");
  const [classification, setClassification] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { role: "bot", text: "Hello! I'm NexoOps AI Assistant. Upload logs or ask me about network issues." }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [keywordCounts, setKeywordCounts] = useState([]);
  const [activeTab, setActiveTab] = useState("logs"); // logs, chat, analytics
  const chatEndRef = useRef(null);

  
  const BACKEND_URL = "http://127.0.0.1:5000";


  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);


  // ---------------- Handle File Upload ----------------
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const content = event.target.result;
        setLogText(content);
        await processLog(content);
      };
      reader.readAsText(file);
    }
  };

  // ---------------- Process Log: Summarize + Classify ----------------
  const processLog = async (text) => {
    if (!text) return;
    setLoading(true);
    try {
      // 1️⃣ Summarize
      const sumRes = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: text }),
      });
      const sumData = await sumRes.json();
      setSummary(sumData.summary);

      // Extract keywords for graph
      const lines = text.split("\n").filter(l => l.trim() !== "");
      const criticalWords = ['error','fail','warning','timeout','critical','fatal','panic','crash','corruption','breach'];
      const counts = {};
      lines.forEach(line => {
        criticalWords.forEach(word => {
          if(line.toLowerCase().includes(word)){
            counts[word] = (counts[word] || 0) + 1;
          }
        });
      });
      const sortedKeywords = Object.entries(counts)
        .map(([key,value]) => ({ keyword: key, count: value }))
        .sort((a,b) => b.count - a.count)
        .slice(0,5);
      setKeywordCounts(sortedKeywords);

      // 2️⃣ Classify Alert
      const classRes = await fetch(`${BACKEND_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: sumData.summary }),
      });
      const classData = await classRes.json();
      setClassification(classData.classification);
    } catch (err) {
      console.error(err);
      alert("Error processing log file");
    } finally {
      setLoading(false);
    }
  };

  // ---------------- Chatbot ----------------
  const handleChat = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage = { role: "user", text: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput("");

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: chatInput }),
      });
      const data = await response.json();
      
      setTimeout(() => {
        setChatMessages(prev => [...prev, { role: "bot", text: data.response }]);
      }, 500);
    } catch (err) {
      console.error(err);
      setChatMessages(prev => [...prev, { 
        role: "bot", 
        text: "Error connecting to backend. Please ensure the server is running." 
      }]);
    }
  };

  // Prepare severity data for bar chart
  const severityData = classification && classification.probabilities
    ? Object.entries(classification.probabilities).map(([key, value]) => ({
        severity: key,
        probability: (value * 100).toFixed(1)
      }))
    : [];

  // Mock network activity data
  const networkActivity = [
    { time: "00:00", traffic: 45, errors: 2 },
    { time: "04:00", traffic: 32, errors: 1 },
    { time: "08:00", traffic: 78, errors: 5 },
    { time: "12:00", traffic: 92, errors: 8 },
    { time: "16:00", traffic: 65, errors: 3 },
    { time: "20:00", traffic: 54, errors: 2 },
  ];

  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Header */}
      <header className="border-b border-cyan-900/50 bg-gradient-to-r from-black via-gray-900 to-black backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-50 animate-pulse"></div>
                <Zap className="w-8 h-8 text-cyan-400 relative" strokeWidth={2.5} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-orange-500">
                  NexoOps
                </h1>
                <p className="text-xs text-cyan-600">Intelligent Network Management System</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 px-4 py-2 bg-cyan-950/30 border border-cyan-800/50 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-cyan-400">SYSTEM ONLINE</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Panel - Navigation & Stats */}
          <div className="lg:col-span-1 space-y-6">
            {/* Navigation Tabs */}
            <motion.div 
              className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl p-4 shadow-2xl"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <h2 className="text-sm text-cyan-400 mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                CONTROL PANEL
              </h2>
              <div className="space-y-2">
                {[
                  { id: "logs", icon: FileText, label: "Log Analysis" },
                  { id: "chat", icon: MessageSquare, label: "ChatOps" },
                  { id: "analytics", icon: BarChart3, label: "Analytics" }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      activeTab === tab.id
                        ? "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400"
                        : "bg-gray-800/50 border border-gray-700/50 text-gray-400 hover:bg-gray-800 hover:text-cyan-400"
                    }`}
                  >
                    <tab.icon className="w-5 h-5" />
                    <span className="font-semibold">{tab.label}</span>
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div 
              className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl p-4 shadow-2xl"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <h2 className="text-sm text-cyan-400 mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                SYSTEM METRICS
              </h2>
              <div className="space-y-4">
                {[
                  { label: "Uptime", value: "99.8%", color: "green" },
                  { label: "Alerts", value: classification ? "1 Active" : "0 Active", color: classification ? "orange" : "cyan" },
                  { label: "Logs Processed", value: logText ? "1" : "0", color: "cyan" }
                ].map((stat, i) => (
                  <div key={i} className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">{stat.label}</span>
                    <span className={`text-${stat.color}-400 font-bold`}>{stat.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Classification Result */}
            {classification && (
              <motion.div 
                className="bg-gradient-to-br from-orange-950/30 to-black border border-orange-500/50 rounded-xl p-4 shadow-2xl"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <h3 className="text-sm text-orange-400 font-bold">ALERT STATUS</h3>
                </div>
                <div className="bg-black/50 rounded-lg p-3 border border-orange-800/30">
                  <div className="text-2xl font-bold text-orange-400 mb-1">
                    {classification.severity}
                  </div>
                  <div className="text-xs text-gray-400">Severity Level Detected</div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-2">
            <AnimatePresence mode="wait">
              {/* Logs Tab */}
              {activeTab === "logs" && (
                <motion.div
                  key="logs"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Upload Section */}
                  <div className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl p-6 shadow-2xl">
                    <h2 className="text-lg text-cyan-400 mb-4 flex items-center gap-2">
                      <Upload className="w-5 h-5" />
                      LOG FILE ANALYZER
                    </h2>
                    
                    <div className="mb-4">
                      <label className="block mb-2 text-sm text-gray-400">Upload Log File</label>
                      <div className="relative">
                        <input
                          type="file"
                          accept=".txt,.log"
                          onChange={handleFileUpload}
                          className="hidden"
                          id="file-upload"
                        />
                        <label
                          htmlFor="file-upload"
                          className="flex items-center justify-center gap-2 w-full p-4 bg-cyan-950/30 border-2 border-dashed border-cyan-800/50 rounded-lg cursor-pointer hover:border-cyan-600 hover:bg-cyan-950/50 transition-all"
                        >
                          <Upload className="w-5 h-5 text-cyan-400" />
                          <span className="text-cyan-400">Click to upload or drag & drop</span>
                        </label>
                      </div>
                    </div>

                    <div>
                      <label className="block mb-2 text-sm text-gray-400">Or paste logs directly</label>
                      <textarea
                        rows="8"
                        value={logText}
                        onChange={(e) => setLogText(e.target.value)}
                        placeholder="2024-11-16 10:23:45 ERROR: Connection timeout on port 8080&#x0a;2024-11-16 10:23:46 WARNING: High memory usage detected..."
                        className="w-full p-4 rounded-lg bg-black border border-cyan-900/50 text-cyan-100 focus:outline-none focus:border-cyan-500 transition font-mono text-sm resize-none"
                      />
                    </div>

                    {loading && (
                      <div className="mt-4 flex items-center gap-3 text-cyan-400">
                        <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                        <span className="text-sm">Processing logs...</span>
                      </div>
                    )}
                  </div>

                  {/* Summary & Results */}
                  {summary && (
                    <motion.div
                      className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl p-6 shadow-2xl"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      <h3 className="text-lg text-cyan-400 mb-4 flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        AI ANALYSIS SUMMARY
                      </h3>
                      <div className="bg-black/50 p-4 rounded-lg border border-cyan-900/30">
                        <pre className="whitespace-pre-wrap text-sm text-cyan-100">{summary}</pre>
                      </div>

                      {/* Severity Probabilities */}
                      {severityData.length > 0 && (
                        <div className="mt-6">
                          <h4 className="text-sm text-cyan-400 mb-3">SEVERITY DISTRIBUTION</h4>
                          <div className="bg-black/50 p-4 rounded-lg border border-cyan-900/30">
                            <ResponsiveContainer width="100%" height={250}>
                              <BarChart data={severityData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#0e7490" opacity={0.3} />
                                <XAxis 
                                  dataKey="severity" 
                                  stroke="#06b6d4" 
                                  style={{ fontSize: '12px' }}
                                />
                                <YAxis stroke="#06b6d4" style={{ fontSize: '12px' }} />
                                <Tooltip 
                                  contentStyle={{ 
                                    backgroundColor: '#000', 
                                    border: '1px solid #06b6d4',
                                    borderRadius: '8px'
                                  }}
                                />
                                <Bar dataKey="probability" fill="#06b6d4" radius={[8, 8, 0, 0]} />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      )}

                      {/* Top Keywords */}
                      {keywordCounts.length > 0 && (
                        <div className="mt-6">
                          <h4 className="text-sm text-cyan-400 mb-3">TOP CRITICAL KEYWORDS</h4>
                          <div className="bg-black/50 p-4 rounded-lg border border-cyan-900/30">
                            <ResponsiveContainer width="100%" height={200}>
                              <BarChart layout="vertical" data={keywordCounts}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#fb923c" opacity={0.2} />
                                <XAxis type="number" stroke="#fb923c" style={{ fontSize: '12px' }} />
                                <YAxis 
                                  type="category" 
                                  dataKey="keyword" 
                                  stroke="#fb923c" 
                                  style={{ fontSize: '12px' }}
                                />
                                <Tooltip 
                                  contentStyle={{ 
                                    backgroundColor: '#000', 
                                    border: '1px solid #fb923c',
                                    borderRadius: '8px'
                                  }}
                                />
                                <Bar dataKey="count" fill="#fb923c" radius={[0, 8, 8, 0]} />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* Chat Tab */}
              {activeTab === "chat" && (
                <motion.div
                  key="chat"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl shadow-2xl flex flex-col"
                  style={{ height: "calc(100vh - 220px)" }}
                >
                  {/* Chat Header */}
                  <div className="p-4 border-b border-cyan-900/50">
                    <div className="flex items-center gap-3">
                      <div className="relative">
                        <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-cyan-700 rounded-full flex items-center justify-center">
                          <Server className="w-6 h-6 text-white" />
                        </div>
                        <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-gray-900"></div>
                      </div>
                      <div>
                        <h3 className="text-cyan-400 font-bold">NexoOps AI Assistant</h3>
                        <p className="text-xs text-gray-500">Network Operations ChatOps</p>
                      </div>
                    </div>
                  </div>

                  {/* Chat Messages */}
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {chatMessages.map((msg, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div className={`flex gap-2 max-w-[80%] ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                          {msg.role === "bot" && (
                            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-cyan-700 rounded-full flex items-center justify-center flex-shrink-0">
                              <Server className="w-5 h-5 text-white" />
                            </div>
                          )}
                          <div
                            className={`p-3 rounded-2xl ${
                              msg.role === "user"
                                ? "bg-cyan-600 text-white rounded-tr-none"
                                : "bg-gray-800 text-cyan-100 rounded-tl-none border border-cyan-900/50"
                            }`}
                          >
                            <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Chat Input */}
                  <div className="p-4 border-t border-cyan-900/50">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyPress={(e) => e.key === "Enter" && handleChat()}
                        placeholder="Ask about network issues, logs, or diagnostics..."
                        className="flex-1 p-3 rounded-lg bg-black border border-cyan-900/50 text-cyan-100 focus:outline-none focus:border-cyan-500 transition text-sm"
                      />
                      <button
                        onClick={handleChat}
                        className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg transition-all flex items-center gap-2 font-bold"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Analytics Tab */}
              {activeTab === "analytics" && (
                <motion.div
                  key="analytics"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-gradient-to-br from-gray-900 to-black border border-cyan-900/50 rounded-xl p-6 shadow-2xl">
                    <h2 className="text-lg text-cyan-400 mb-4 flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      NETWORK ACTIVITY
                    </h2>
                    <div className="bg-black/50 p-4 rounded-lg border border-cyan-900/30">
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={networkActivity}>
                          <defs>
                            <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorErrors" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#fb923c" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="#fb923c" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#0e7490" opacity={0.3} />
                          <XAxis dataKey="time" stroke="#06b6d4" style={{ fontSize: '12px' }} />
                          <YAxis stroke="#06b6d4" style={{ fontSize: '12px' }} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: '#000', 
                              border: '1px solid #06b6d4',
                              borderRadius: '8px'
                            }}
                          />
                          <Legend />
                          <Area 
                            type="monotone" 
                            dataKey="traffic" 
                            stroke="#06b6d4" 
                            fillOpacity={1} 
                            fill="url(#colorTraffic)" 
                          />
                          <Area 
                            type="monotone" 
                            dataKey="errors" 
                            stroke="#fb923c" 
                            fillOpacity={1} 
                            fill="url(#colorErrors)" 
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;