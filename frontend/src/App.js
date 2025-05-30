import React, { useState } from "react";


function App() {
  const [query, setQuery] = useState("");
  const [responses, setResponses] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  const handleSubmit = async (e) => {
    e.preventDefault();


    if (!query.trim()) {
      setError("Please enter a query.");
      return;
    }


    setError("");
    setLoading(true);
    setResponses(null);


    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });


      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }


      const data = await res.json();
      setResponses(data);
    } catch (err) {
      setError(`Failed to get response: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
            🎯 Support Ticket Resolution System
          </h1>


          <form onSubmit={handleSubmit} className="mb-6">
            <div className="flex gap-4">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your support query..."
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? "🔄 Processing..." : "Submit"}
              </button>
            </div>
          </form>


          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}


          {responses && (
            <div className="space-y-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-800 mb-2">
                  📄 Knowledge Base Response
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">
                  {responses.pdf_response}
                </p>
              </div>


              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-2">
                  🗃️ Historical Tickets Response
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">
                  {responses.sql_response}
                </p>
              </div>


              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-purple-800 mb-2">
                  🧠 Final Combined Solution
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">
                  {responses.final_response}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


export default App;

