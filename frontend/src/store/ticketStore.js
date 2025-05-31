import { create } from 'zustand';
const useTicketStore = create((set) => ({
// Search state
searchType: 'text', // 'text' or 'ticket'
searchQuery: '',
selectedTicket: null,
// Filter state
dateRange: 'Last 30 days',
customerFilter: '',
engineerFilter: '',
// Results state
searchResults: [],
isLoading: false,
// Actions
setSearchType: (type) => set({ searchType: type }),
setSearchQuery: (query) => set({ searchQuery: query }),
setSelectedTicket: (ticket) => set({ selectedTicket: ticket }),
setDateRange: (range) => set({ dateRange: range }),
setCustomerFilter: (filter) => set({ customerFilter: filter }),
setEngineerFilter: (filter) => set({ engineerFilter: filter }),
setSearchResults: (results) => set({ searchResults: results }),
setLoading: (loading) => set({ isLoading: loading }),
// Search function
performSearch: async () => {
set({ isLoading: true });
// Your backend API call would go here
// For now, just simulate loading
setTimeout(() => {
set({ isLoading: false, searchResults: [] });
}, 1000);
}
}));
export default useTicketStore;