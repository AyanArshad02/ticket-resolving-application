import React from 'react';
import useTicketStore from '../store/ticketStore';
 
const FilterSection = () => {
  const {
    dateRange,
    customerFilter,
    engineerFilter,
    setDateRange,
    setCustomerFilter,
    setEngineerFilter
  } = useTicketStore();
 
  return (
<div className="mb-6">
<h3 className="text-lg font-semibold text-gray-800 mb-4">Filter Results</h3>
      {/* Date Range */}
<div className="mb-4">
<label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
<div className="relative">
<select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none bg-white"
>
<option>Last 30 days</option>
<option>Last 7 days</option>
<option>Last 90 days</option>
<option>Custom range</option>
</select>
<div className="absolute right-2 top-2 pointer-events-none">
            📅
</div>
</div>
</div>
 
      {/* Customer Filter */}
<div className="mb-4">
<label className="block text-sm font-medium text-gray-700 mb-2">Customer</label>
<div className="relative">
<input
            type="text"
            value={customerFilter}
            onChange={(e) => setCustomerFilter(e.target.value)}
            placeholder="Filter by customer"
            className="w-full p-2 pl-8 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
<div className="absolute left-2 top-2 text-gray-400">
            👤
</div>
</div>
</div>
 
      {/* Support Engineer Filter */}
<div className="mb-4">
<label className="block text-sm font-medium text-gray-700 mb-2">Support Engineer</label>
<div className="relative">
<input
            type="text"
            value={engineerFilter}
            onChange={(e) => setEngineerFilter(e.target.value)}
            placeholder="Filter by engineer"
            className="w-full p-2 pl-8 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
<div className="absolute left-2 top-2 text-gray-400">
            👤
</div>
</div>
</div>
</div>
  );
};
 
export default FilterSection;