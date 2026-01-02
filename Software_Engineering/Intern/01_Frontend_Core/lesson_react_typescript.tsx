/**
 * ================================================================
 * SE INTERN - LESSON 3: REACT + TYPESCRIPT BASICS
 * ================================================================
 */

import React, { useState, useEffect } from 'react';

// --- 1. THEORY ---
/**
 * 1. React Core Concepts:
 *    - Components: Reusable UI pieces
 *    - Props: Data passed from parent to child
 *    - State: Component's internal data
 * 
 * 2. Hooks:
 *    - useState: Manage local state
 *    - useEffect: Side effects (fetch, subscriptions)
 * 
 * 3. TypeScript Benefits:
 *    - Type safety at compile time
 *    - Better IDE autocomplete
 *    - Self-documenting code
 * 
 * 4. Component Patterns:
 *    - Presentational: UI only, receives props
 *    - Container: Logic, passes data down
 */

// --- 2. CODE SAMPLE ---

// ========== TYPE DEFINITIONS ==========
interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
}

interface ButtonProps {
    label: string;
    variant?: 'primary' | 'secondary' | 'danger';
    disabled?: boolean;
    onClick: () => void;
}

// ========== FUNCTIONAL COMPONENT ==========
const Button: React.FC<ButtonProps> = ({
    label,
    variant = 'primary',
    disabled = false,
    onClick
}) => {
    const baseStyle = "px-4 py-2 rounded font-medium transition";
    const variants = {
        primary: "bg-blue-500 text-white hover:bg-blue-600",
        secondary: "bg-gray-200 text-gray-800 hover:bg-gray-300",
        danger: "bg-red-500 text-white hover:bg-red-600"
    };

    return (
        <button
            className={`${baseStyle} ${variants[variant as keyof typeof variants]}`}
            disabled={disabled}
            onClick={onClick}
        >
            {label}
        </button>
    );
};

// ========== STATE MANAGEMENT ==========
const Counter: React.FC = () => {
    const [count, setCount] = useState<number>(0);

    const increment = () => setCount(prev => prev + 1);
    const decrement = () => setCount(prev => prev - 1);
    const reset = () => setCount(0);

    return (
        <div className="p-4 border rounded">
            <h2 className="text-xl mb-4">Counter: {count}</h2>
            <div className="flex gap-2">
                <Button label="-" variant="secondary" onClick={decrement} />
                <Button label="Reset" variant="danger" onClick={reset} />
                <Button label="+" variant="primary" onClick={increment} />
            </div>
        </div>
    );
};

// ========== USEEFFECT EXAMPLE ==========
const UserList: React.FC = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                setLoading(true);
                const response = await fetch('https://jsonplaceholder.typicode.com/users');
                if (!response.ok) throw new Error('Failed to fetch');
                const data = await response.json();
                setUsers(data.slice(0, 5));
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setLoading(false);
            }
        };

        fetchUsers();
    }, []); // Empty dependency array = run once on mount

    if (loading) return <div>Loading...</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;

    return (
        <ul className="space-y-2">
            {users.map(user => (
                <li key={user.id} className="p-2 border rounded">
                    {user.name} - {user.email}
                </li>
            ))}
        </ul>
    );
};

// ========== FORM HANDLING ==========
interface FormData {
    email: string;
    password: string;
}

const LoginForm: React.FC = () => {
    const [formData, setFormData] = useState<FormData>({
        email: '',
        password: ''
    });
    const [errors, setErrors] = useState<Partial<FormData>>({});

    const validate = (): boolean => {
        const newErrors: Partial<FormData> = {};

        if (!formData.email.includes('@')) {
            newErrors.email = 'Invalid email format';
        }
        if (formData.password.length < 6) {
            newErrors.password = 'Password must be at least 6 characters';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (validate()) {
            console.log('Form submitted:', formData);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name as keyof FormData]: value }));
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4 max-w-md">
            <div>
                <input
                    type="email"
                    name="email"
                    placeholder="Email"
                    value={formData.email}
                    onChange={handleChange}
                    className="w-full p-2 border rounded"
                />
                {errors.email && (
                    <p className="text-red-500 text-sm">{errors.email}</p>
                )}
            </div>
            <div>
                <input
                    type="password"
                    name="password"
                    placeholder="Password"
                    value={formData.password}
                    onChange={handleChange}
                    className="w-full p-2 border rounded"
                />
                {errors.password && (
                    <p className="text-red-500 text-sm">{errors.password}</p>
                )}
            </div>
            <Button label="Login" variant="primary" onClick={() => { }} />
        </form>
    );
};

// --- 3. EXERCISES ---
/**
 * EXERCISE 1: Create a TodoList component
 *            - Add, delete, toggle completion
 *            - Filter by: all, active, completed
 *            - Persist to localStorage
 * 
 * EXERCISE 2: Build an API data table
 *            - Fetch and display data
 *            - Add sorting by columns
 *            - Add pagination
 * 
 * EXERCISE 3: Create a theme switcher
 *            - Toggle between light/dark mode
 *            - Use Context API
 *            - Persist preference
 * 
 * EXERCISE 4: Build a multi-step form
 *            - 3 steps: Personal Info, Address, Review
 *            - Navigation: Next, Back, Submit
 *            - Validate each step
 */

// Export components
export { Button, Counter, UserList, LoginForm };
