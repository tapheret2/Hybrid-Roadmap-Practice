/**
 * ================================================================
 * SE JUNIOR - FRONTEND ADVANCED: REACT ECOSYSTEM
 * ================================================================
 * 
 * React Hooks nâng cao, State Management, và Data Fetching
 */

import React, { useState, useEffect, useContext, useCallback, useMemo, useReducer, createContext } from 'react';

// --- 1. LÝ THUYẾT (THEORY) ---
/**
 * 1. React Hooks Deep Dive:
 *    - useState: Local state
 *    - useEffect: Side effects (fetch, subscriptions)
 *    - useContext: Share state across components
 *    - useCallback: Memoize functions
 *    - useMemo: Memoize computed values
 *    - useReducer: Complex state logic
 * 
 * 2. State Management:
 *    - Context API: Built-in, simple sharing
 *    - Redux Toolkit: Global state, time travel
 *    - Zustand: Lightweight alternative
 * 
 * 3. Data Fetching:
 *    - React Query/TanStack Query: Caching, refetching
 *    - SWR: Stale-while-revalidate pattern
 * 
 * 4. Performance:
 *    - React.memo: Prevent unnecessary re-renders
 *    - useMemo/useCallback: Memoization
 *    - Code splitting với React.lazy
 */

// --- 2. CODE MẪU (CODE SAMPLE) ---

// ========== CUSTOM HOOKS ==========

// Custom hook for API calls
function useFetch<T>(url: string) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const controller = new AbortController();

        async function fetchData() {
            try {
                setLoading(true);
                const res = await fetch(url, { signal: controller.signal });
                const json = await res.json();
                setData(json);
            } catch (err) {
                if (err instanceof Error && err.name !== 'AbortError') {
                    setError(err);
                }
            } finally {
                setLoading(false);
            }
        }

        fetchData();

        return () => controller.abort(); // Cleanup
    }, [url]);

    return { data, loading, error };
}

// Custom hook for localStorage
function useLocalStorage<T>(key: string, initialValue: T) {
    const [value, setValue] = useState<T>(() => {
        const stored = localStorage.getItem(key);
        return stored ? JSON.parse(stored) : initialValue;
    });

    useEffect(() => {
        localStorage.setItem(key, JSON.stringify(value));
    }, [key, value]);

    return [value, setValue] as const;
}

// Custom hook for debounce
function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);

    return debouncedValue;
}

// ========== CONTEXT + REDUCER PATTERN ==========

interface AuthState {
    user: { id: string; name: string } | null;
    isAuthenticated: boolean;
    loading: boolean;
}

type AuthAction =
    | { type: 'LOGIN_START' }
    | { type: 'LOGIN_SUCCESS'; payload: { id: string; name: string } }
    | { type: 'LOGIN_FAILURE' }
    | { type: 'LOGOUT' };

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
    switch (action.type) {
        case 'LOGIN_START':
            return { ...state, loading: true };
        case 'LOGIN_SUCCESS':
            return { user: action.payload, isAuthenticated: true, loading: false };
        case 'LOGIN_FAILURE':
            return { user: null, isAuthenticated: false, loading: false };
        case 'LOGOUT':
            return { user: null, isAuthenticated: false, loading: false };
        default:
            return state;
    }
};

const AuthContext = createContext<{
    state: AuthState;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
} | null>(null);

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [state, dispatch] = useReducer(authReducer, {
        user: null,
        isAuthenticated: false,
        loading: false,
    });

    const login = async (email: string, password: string) => {
        dispatch({ type: 'LOGIN_START' });
        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            dispatch({ type: 'LOGIN_SUCCESS', payload: { id: '1', name: 'User' } });
        } catch {
            dispatch({ type: 'LOGIN_FAILURE' });
        }
    };

    const logout = () => dispatch({ type: 'LOGOUT' });

    return (
        <AuthContext.Provider value={{ state, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

// ========== PERFORMANCE OPTIMIZATION ==========

interface ExpensiveListProps {
    items: string[];
    onItemClick: (item: string) => void;
}

// Memoized component
const ExpensiveList = React.memo(({ items, onItemClick }: ExpensiveListProps) => {
    console.log('ExpensiveList rendered');
    return (
        <ul>
            {items.map(item => (
                <li key={item} onClick={() => onItemClick(item)}>{item}</li>
            ))}
        </ul>
    );
});

// Parent with memoization
const ParentComponent: React.FC = () => {
    const [count, setCount] = useState(0);
    const [items] = useState(['A', 'B', 'C']);

    // useCallback: Memoize function
    const handleClick = useCallback((item: string) => {
        console.log('Clicked:', item);
    }, []);

    // useMemo: Memoize expensive computation
    const expensiveValue = useMemo(() => {
        console.log('Computing expensive value...');
        return items.reduce((acc, item) => acc + item.length, 0);
    }, [items]);

    return (
        <div>
            <button onClick={() => setCount(c => c + 1)}>Count: {count}</button>
            <p>Expensive value: {expensiveValue}</p>
            <ExpensiveList items={items} onItemClick={handleClick} />
        </div>
    );
};

// --- 3. BÀI TẬP (EXERCISE) ---
/**
 * BÀI 1: Tạo custom hook useAsync:
 *        - Handle loading, error, data states
 *        - Support retry functionality
 *        - Cancel on unmount
 * 
 * BÀI 2: Implement shopping cart với useReducer:
 *        - Add, remove, update quantity
 *        - Calculate total
 *        - Persist to localStorage
 * 
 * BÀI 3: Create infinite scroll với React Query:
 *        - Fetch paginated data
 *        - Load more on scroll
 *        - Show loading indicator
 * 
 * BÀI 4: Optimize large list với virtualization:
 *        - Use react-window or react-virtualized
 *        - Render only visible items
 */

export { useFetch, useLocalStorage, useDebounce, AuthProvider, ParentComponent };
