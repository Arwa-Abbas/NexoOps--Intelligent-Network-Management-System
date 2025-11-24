import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from summarizer import summarize_log
from alert_classifier import classify_log, train_alert_model
from chatbot import chatbot_response, get_chatbot
import threading

app = Flask(__name__)
CORS(app)

# Train model on startup
model_thread = threading.Thread(target=train_alert_model, daemon=True)
model_thread.start()

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "NexoOps Backend API",
        "version": "1.0.0",
        "endpoints": {
            "log_analysis": ["/summarize", "/classify"],
            "chatbot": ["/chat"],
            "network": ["/network/status", "/network/alerts", "/network/speed-test",
                       "/network/interfaces", "/network/connections", "/network/bandwidth",
                       "/network/ping", "/network/port-check", "/network/health"]
        }
    })


# ==================== LOG ANALYSIS ENDPOINTS ====================

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize log text"""
    try:
        data = request.get_json()
        text = data.get("log_text", "")
        
        if not text:
            return jsonify({"error": "No log text provided"}), 400
        
        n_sentences = data.get("n_sentences", 5)
        summary = summarize_log(text, n_sentences=n_sentences)
        
        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/classify', methods=['POST'])
def classify():
    """Classify log severity"""
    try:
        data = request.get_json()
        text = data.get("log_text", "")
        
        if not text:
            return jsonify({"error": "No log text provided"}), 400
        
        result = classify_log(text)
        
        return jsonify({
            "classification": result,
            "timestamp": request.args.get('timestamp', None)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Complete log analysis (summary + classification)"""
    try:
        data = request.get_json()
        text = data.get("log_text", "")
        
        if not text:
            return jsonify({"error": "No log text provided"}), 400
        
        summary = summarize_log(text)
        classification = classify_log(text)
        
        return jsonify({
            "summary": summary,
            "classification": classification,
            "original_length": len(text)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== CHATBOT ENDPOINTS ====================

@app.route('/chat', methods=['POST'])
def chat():
    """Chat with network assistant"""
    try:
        data = request.get_json()
        message = data.get("message", "")
        log_context = data.get("log_context", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        response = chatbot_response(message, log_context)
        
        return jsonify({
            "response": response,
            "timestamp": request.args.get('timestamp', None)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== NETWORK MONITORING ENDPOINTS ====================

@app.route('/network/status', methods=['GET'])
def network_status():
    """Get current network status"""
    try:
        bot = get_chatbot()
        
        stats = bot.ops.get_network_stats()  
        bandwidth = bot.ops.get_bandwidth()  
        
        return jsonify({
            "stats": stats,
            "bandwidth": bandwidth,
            "status": "healthy" if "error" not in stats else "error"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/alerts', methods=['GET'])
def network_alerts():
    """Get network alerts"""
    try:
        bot = get_chatbot()
        hours = int(request.args.get('hours', 24))
        
        current_alerts = bot.ops.check_alerts()
        recent_alerts = bot.ops.get_recent_alerts(hours=hours)
        
        return jsonify({
            "alerts": current_alerts + recent_alerts,
            "count": len(current_alerts + recent_alerts),
            "time_window_hours": hours
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/speed-test', methods=['POST'])
def speed_test():
    """Run internet speed test"""
    try:
        bot = get_chatbot()
        result = bot.ops.speed_test() 
        
        return jsonify({
            "speed_test": result,
            "status": "success" if "error" not in result else "failed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/interfaces', methods=['GET'])
def network_interfaces():
    """Get network interfaces"""
    try:
        bot = get_chatbot()
        interfaces = bot.ops.get_interfaces() 
        
        return jsonify({
            "interfaces": interfaces,
            "count": len(interfaces) if isinstance(interfaces, list) else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/connections', methods=['GET'])
def network_connections():
    """Get active network connections"""
    try:
        bot = get_chatbot()
        limit = int(request.args.get('limit', 20))
        connections = bot.ops.get_connections(limit=limit)  
        
        return jsonify({
            "connections": connections,
            "count": len(connections) if isinstance(connections, list) else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/bandwidth', methods=['GET'])
def network_bandwidth():
    """Get current bandwidth usage"""
    try:
        bot = get_chatbot()
        bandwidth = bot.ops.get_bandwidth()  
        
        return jsonify({
            "bandwidth": bandwidth,
            "status": "success" if "error" not in bandwidth else "failed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/ping', methods=['POST'])
def network_ping():
    """Ping a host"""
    try:
        data = request.get_json()
        host = data.get("host", "")
        
        if not host:
            return jsonify({"error": "No host provided"}), 400
        
        bot = get_chatbot()
        result = bot.ops.ping(host)  
        
        return jsonify({
            "ping_result": result,
            "host": host
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/port-check', methods=['POST'])
def port_check():
    """Check if a port is open"""
    try:
        data = request.get_json()
        host = data.get("host", "")
        port = data.get("port", None)
        
        if not host or port is None:
            return jsonify({"error": "Host and port required"}), 400
        
        bot = get_chatbot()
        result = bot.ops.check_port(host, int(port))  
        
        return jsonify({
            "port_check": result,
            "host": host,
            "port": port
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/health', methods=['GET'])
def system_health():
    """Get system health metrics"""
    try:
        bot = get_chatbot()
        health = bot.ops.system_health()  
        
        return jsonify({
            "health": health,
            "status": health.get('overall', 'unknown')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/network/history', methods=['GET'])
def network_history():
    """Get network statistics history"""
    try:
        bot = get_chatbot()
        stats = bot.ops.get_network_stats()
        
        return jsonify({
            "current_stats": stats,
            "message": "Real-time stats only (history not implemented)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== UTILITY ENDPOINTS ====================

@app.route('/train-model', methods=['POST'])
def train_model():
    """Manually trigger model training"""
    try:
        def train_in_background():
            train_alert_model()
        
        thread = threading.Thread(target=train_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            "message": "Model training started in background",
            "status": "training"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def api_stats():
    """Get API statistics"""
    try:
        bot = get_chatbot()
        
        return jsonify({
            "alerts_count": len(bot.ops.alerts),
            "logs_count": len(bot.ops.logs.logs),
            "system_status": "operational"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("NexoOps Backend API Server")
    print("=" * 60)
    print("\nAvailable Endpoints:")
    print("\nLog Analysis:")
    print("  POST /summarize        - Summarize log text")
    print("  POST /classify         - Classify log severity")
    print("  POST /analyze          - Complete analysis")
    print("\nChatbot:")
    print("  POST /chat             - Chat with assistant")
    print("\nNetwork Monitoring:")
    print("  GET  /network/status   - Current network status")
    print("  GET  /network/alerts   - Recent alerts")
    print("  POST /network/speed-test - Run speed test")
    print("  GET  /network/interfaces - List interfaces")
    print("  GET  /network/connections - Active connections")
    print("  GET  /network/bandwidth - Bandwidth usage")
    print("  POST /network/ping     - Ping a host")
    print("  POST /network/port-check - Check port")
    print("  GET  /network/health   - System health")
    print("  GET  /network/history  - Network history")
    print("\nUtility:")
    print("  POST /train-model      - Train ML model")
    print("  GET  /stats            - API statistics")
    print("=" * 60)
    print("\nStarting server on http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)