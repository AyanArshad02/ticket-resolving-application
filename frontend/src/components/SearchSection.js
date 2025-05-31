import React from 'react';

import useTicketStore from '../store/ticketStore';
 
const SearchSection = () => {

  const {

    searchType,

    searchQuery,

    setSearchType,

    setSearchQuery,

    performSearch

  } = useTicketStore();
 
  const handleSearch = () => {

    performSearch();

  };
 
  return (
<div className="mb-6">
<h2 className="text-lg font-semibold text-gray-800 mb-4">Search Similar Tickets</h2>

      {/* Search Type Tabs */}
<div className="flex mb-4">
<button

          onClick={() => setSearchType('text')}

          className={`px-4 py-2 rounded-l-lg border ${

            searchType === 'text'

              ? 'bg-blue-100 border-blue-300 text-blue-700'

              : 'bg-gray-100 border-gray-300 text-gray-600'

          }`}
>

          Search by Text
</button>
<button

          onClick={() => setSearchType('ticket')}

          className={`px-4 py-2 rounded-r-lg border-t border-r border-b ${

            searchType === 'ticket'

              ? 'bg-blue-100 border-blue-300 text-blue-700'

              : 'bg-gray-100 border-gray-300 text-gray-600'

          }`}
>

          Search by Ticket #
</button>
</div>
 
      {/* Search Input */}
<div className="flex gap-2">
<input

          type="text"

          value={searchQuery}

          onChange={(e) => setSearchQuery(e.target.value)}

          placeholder={searchType === 'text' ? "Describe the issue..." : "Enter ticket number..."}

          className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"

          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}

        />
<button

          onClick={handleSearch}

          className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-dark"
>

          🔍
</button>
</div>
</div>

  );

};
 
export default SearchSection;