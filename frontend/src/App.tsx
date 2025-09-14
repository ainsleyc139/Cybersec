import { useState, useEffect } from 'react'
import './App.css'

interface ApiResponse {
  message: string;
  version: string;
  endpoints: {
    health: string;
    api: string;
  };
}

interface HealthResponse {
  status: string;
  timestamp: string;
  uptime: number;
}

function App() {
  const [apiStatus, setApiStatus] = useState<'loading' | 'connected' | 'error'>('loading');
  const [apiData, setApiData] = useState<ApiResponse | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);

  useEffect(() => {
    checkApiConnection();
  }, []);

  const checkApiConnection = async () => {
    try {
      // Test API connection
      const apiResponse = await fetch('http://localhost:3001/api');
      const apiResult = await apiResponse.json();
      setApiData(apiResult);

      // Test health endpoint
      const healthResponse = await fetch('http://localhost:3001/health');
      const healthResult = await healthResponse.json();
      setHealthData(healthResult);

      setApiStatus('connected');
    } catch (error) {
      console.error('API connection failed:', error);
      setApiStatus('error');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-blue-400 mb-4">
            🛡️ Cybersec Project
          </h1>
          <p className="text-xl text-gray-300">
            Educational Cybersecurity Platform
          </p>
        </header>

        {/* API Status */}
        <div className="mb-8">
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <h2 className="text-2xl font-semibold mb-4 text-green-400">
              🔗 Backend Connection Status
            </h2>
            <div className="flex items-center space-x-4">
              <div className={`w-4 h-4 rounded-full ${
                apiStatus === 'connected' ? 'bg-green-500' : 
                apiStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
              }`}></div>
              <span className="text-lg">
                {apiStatus === 'connected' ? 'Connected' : 
                 apiStatus === 'error' ? 'Disconnected' : 'Connecting...'}
              </span>
              <button 
                onClick={checkApiConnection}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* API Data */}
        {apiData && (
          <div className="mb-8">
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                📊 API Information
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <strong>Message:</strong> {apiData.message}
                </div>
                <div>
                  <strong>Version:</strong> {apiData.version}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Health Data */}
        {healthData && (
          <div className="mb-8">
            <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
              <h3 className="text-xl font-semibold mb-4 text-green-400">
                ❤️ Server Health
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <strong>Status:</strong> {healthData.status}
                </div>
                <div>
                  <strong>Uptime:</strong> {Math.floor(healthData.uptime)} seconds
                </div>
                <div>
                  <strong>Last Check:</strong> {new Date(healthData.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-purple-400">
              🔐 Authentication
            </h3>
            <p className="text-gray-300">
              Secure user authentication with JWT tokens and password hashing.
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-red-400">
              🔍 Vulnerability Scanner
            </h3>
            <p className="text-gray-300">
              Scan for common security vulnerabilities in web applications.
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-yellow-400">
              📈 Security Analytics
            </h3>
            <p className="text-gray-300">
              Monitor and analyze security events and patterns.
            </p>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-gray-500">
          <p>Built with React + TypeScript (Frontend) and Node.js + Express (Backend)</p>
          <p className="mt-2">🎓 Educational Cybersecurity Project</p>
        </footer>
      </div>
    </div>
  )
}

export default App
