import React, { useState } from 'react';
import axios from 'axios';

export default function TicketSearchUI() {
  const [searchMode, setSearchMode] = useState('text');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);

  const handleSearch = async () => {
    try {
      const res = await axios.post('http://127.0.0.1:8000/query', {
        query: query
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <header className="bg-blue-700 text-white text-xl font-semibold p-4">
        PTS Resolution Enhancement System
      </header>

      <div className="flex flex-1">
        {/* Left Sidebar */}
        <div className="w-1/4 bg-gray-100 p-4 space-y-4">
          <div className="flex gap-2">
            <button
              onClick={() => setSearchMode('text')}
              className={`p-2 flex-1 rounded ${searchMode === 'text' ? 'bg-blue-600 text-white' : 'bg-white border'}`}
            >
              Search by Text
            </button>
            <button
              onClick={() => setSearchMode('ticket')}
              className={`p-2 flex-1 rounded ${searchMode === 'ticket' ? 'bg-blue-600 text-white' : 'bg-white border'}`}
            >
              Search by Ticket #
            </button>
          </div>

          <div className="flex items-center border rounded p-2">
            <input
              type="text"
              placeholder="Describe the issue..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 outline-none"
            />
            <button onClick={handleSearch} className="ml-2 text-blue-600">
              🔍
            </button>
          </div>

          <div className="space-y-2">
            <div>
              <label className="block text-sm font-semibold">Date Range</label>
              <select className="w-full border p-2 rounded">
                <option>Last 30 days</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold">Customer</label>
              <input type="text" placeholder="Filter by customer" className="w-full border p-2 rounded" />
            </div>
            <div>
              <label className="block text-sm font-semibold">Support Engineer</label>
              <input type="text" placeholder="Filter by engineer" className="w-full border p-2 rounded" />
            </div>
          </div>
        </div>

        {/* Right Display Panel */}
        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          {result ? (
            <div className="space-y-4">
              {/* PDF Response */}
              <div className="bg-white p-4 shadow rounded">
                <h2 className="font-bold text-lg mb-2">PDF Response</h2>
                <p>{result.pdf_response}</p>
              </div>

              {/* SQL Response (Scrollable Table) */}
              <div className="bg-white p-4 shadow rounded">
                <h2 className="font-bold text-lg mb-2">SQL Response</h2>
                <div className="overflow-x-auto max-h-64 overflow-y-auto">
                  <table className="table-auto w-full border">
                    <thead>
                      <tr>
                        {result.sql_response.length > 0 &&
                          Object.keys(
                            result.sql_response[0]
                              .split('\n')
                              .reduce((acc, line) => {
                                const [key, ...rest] = line.split(':');
                                if (key && rest.length) acc[key.trim()] = rest.join(':').trim();
                                return acc;
                              }, {})
                          ).map((key) => (
                            <th key={key} className="px-4 py-2 border-b bg-gray-100 text-left">
                              {key}
                            </th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.sql_response.map((ticketStr, idx) => {
                        const ticketObj = ticketStr.split('\n').reduce((acc, line) => {
                          const [key, ...rest] = line.split(':');
                          if (key && rest.length) acc[key.trim()] = rest.join(':').trim();
                          return acc;
                        }, {});

                        return (
                          <tr key={idx}>
                            {Object.values(ticketObj).map((val, i) => (
                              <td key={i} className="px-4 py-2 border-b">
                                {val}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Final Response */}
              <div className="bg-white p-4 shadow rounded">
                <h2 className="font-bold text-lg mb-2">Final Response</h2>
                <p>{result.final_response}</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              Select a ticket from the search results to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}




// import React, { useState } from 'react';
// import axios from 'axios';

// export default function TicketSearchUI() {
//   const [searchMode, setSearchMode] = useState('text');
//   const [query, setQuery] = useState('');
//   const [result, setResult] = useState(null);

//   const handleSearch = async () => {
//     try {
//       const res = await axios.post('http://127.0.0.1:8000/query', {
//         query: query
//       });
//       setResult(res.data);
//     } catch (err) {
//       console.error(err);
//     }
//   };
  

//   return (
//     <div className="h-screen flex flex-col">
//       <header className="bg-blue-700 text-white text-xl font-semibold p-4">
//         PTS Resolution Enhancement System
//       </header>

//       <div className="flex flex-1">
//         {/* Left Sidebar */}
//         <div className="w-1/4 bg-gray-100 p-4 space-y-4">
//           <div className="flex gap-2">
//             <button
//               onClick={() => setSearchMode('text')}
//               className={`p-2 flex-1 rounded ${searchMode === 'text' ? 'bg-blue-600 text-white' : 'bg-white border'}`}
//             >
//               Search by Text
//             </button>
//             <button
//               onClick={() => setSearchMode('ticket')}
//               className={`p-2 flex-1 rounded ${searchMode === 'ticket' ? 'bg-blue-600 text-white' : 'bg-white border'}`}
//             >
//               Search by Ticket #
//             </button>
//           </div>

//           <div className="flex items-center border rounded p-2">
//             <input
//               type="text"
//               placeholder="Describe the issue..."
//               value={query}
//               onChange={(e) => setQuery(e.target.value)}
//               className="flex-1 outline-none"
//             />
//             <button onClick={handleSearch} className="ml-2 text-blue-600">
//               🔍
//             </button>
//           </div>

//           <div className="space-y-2">
//             <div>
//               <label className="block text-sm font-semibold">Date Range</label>
//               <select className="w-full border p-2 rounded">
//                 <option>Last 30 days</option>
//               </select>
//             </div>
//             <div>
//               <label className="block text-sm font-semibold">Customer</label>
//               <input type="text" placeholder="Filter by customer" className="w-full border p-2 rounded" />
//             </div>
//             <div>
//               <label className="block text-sm font-semibold">Support Engineer</label>
//               <input type="text" placeholder="Filter by engineer" className="w-full border p-2 rounded" />
//             </div>
//           </div>
//         </div>

//         {/* Right Display Panel */}
//         <div className="flex-1 p-6 overflow-y-auto space-y-4">
//           {result ? (
//             <div className="space-y-4">
//               {/* PDF Response */}
//               <div className="bg-white p-4 shadow rounded">
//                 <h2 className="font-bold text-lg mb-2">PDF Response</h2>
//                 <p>{result.pdf_response}</p>
//               </div>

//               {/* SQL Response (Scrollable Table) */}
//               <div className="bg-white p-4 shadow rounded">
//                 <h2 className="font-bold text-lg mb-2">SQL Response</h2>
//                 <div className="overflow-x-auto max-h-64 overflow-y-auto">
//                   <table className="table-auto w-full border">
//                     <thead>
//                       <tr>
//                         {Object.keys(result.sql_response[0] || {}).map((key) => (
//                           <th key={key} className="px-4 py-2 border-b bg-gray-100 text-left">{key}</th>
//                         ))}
//                       </tr>
//                     </thead>
//                     <tbody>
//                       {result.sql_response.map((row, idx) => (
//                         <tr key={idx}>
//                           {Object.values(row).map((val, i) => (
//                             <td key={i} className="px-4 py-2 border-b">{val}</td>
//                           ))}
//                         </tr>
//                       ))}
//                     </tbody>
//                   </table>
//                 </div>
//               </div>

//               {/* Final Response */}
//               <div className="bg-white p-4 shadow rounded">
//                 <h2 className="font-bold text-lg mb-2">Final Response</h2>
//                 <p>{result.final_response}</p>
//               </div>
//             </div>
//           ) : (
//             <div className="flex items-center justify-center h-full text-gray-500">
//               Select a ticket from the search results to view details
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }
