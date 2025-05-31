import React from 'react';

import useTicketStore from '../store/ticketStore';
 
const SearchResults = () => {

  const { searchResults, isLoading, selectedTicket, setSelectedTicket } = useTicketStore();
 
  if (isLoading) {

    return (
<div className="bg-white rounded-lg border border-gray-200 p-8">
<div className="text-center">
<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
<p className="text-gray-600">Searching tickets...</p>
</div>
</div>

    );

  }
 
  if (searchResults.length === 0) {

    return (
<div className="bg-white rounded-lg border border-gray-200 p-8">
<div className="text-center">
<div className="w-16 h-16 mx-auto mb-4 text-gray-300">

            🔍
</div>
<p className="text-gray-500">Select a ticket from the search results to view details</p>
</div>
</div>

    );

  }
 
  return (
<div className="bg-white rounded-lg border border-gray-200">
<div className="divide-y divide-gray-200">

        {searchResults.map((ticket) => (
<div

            key={ticket.id}

            onClick={() => setSelectedTicket(ticket)}

            className={`p-4 cursor-pointer hover:bg-gray-50 ${

              selectedTicket?.id === ticket.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''

            }`}
>
<div className="flex justify-between items-start">
<div>
<h4 className="font-medium text-gray-900">{ticket.title}</h4>
<p className="text-sm text-gray-600 mt-1">{ticket.description}</p>
<div className="flex gap-2 mt-2">
<span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">

                    {ticket.status}
</span>
<span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">

                    {ticket.priority}
</span>
</div>
</div>
<span className="text-sm text-gray-500">{ticket.date}</span>
</div>
</div>

        ))}
</div>
</div>

  );

};
 
export default SearchResults;