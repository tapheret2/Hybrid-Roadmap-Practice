/**
 * ================================================================
 * SE INTERN - BÀI 3: REACT + TYPESCRIPT BASICS
 * ================================================================
 * 
 * Để chạy được file này, bạn cần tạo project React:
 * npx create-react-app my-app --template typescript
 * hoặc: npm create vite@latest my-app -- --template react-ts
 */

import React, { useState, useEffect } from 'react';

// --- 1. LÝ THUYẾT (THEORY) ---
/**
 * 1. Component: Khối xây dựng UI, có thể tái sử dụng
 * 2. Props: Dữ liệu truyền từ parent → child (read-only)
 * 3. State: Dữ liệu nội bộ của component, thay đổi → re-render
 * 4. useState: Hook để quản lý state
 * 5. useEffect: Hook để side effects (fetch data, subscriptions)
 * 6. TypeScript: Thêm type safety cho JavaScript
 */

// --- 2. CODE MẪU (CODE SAMPLE) ---

// Định nghĩa Type/Interface
interface User {
    id: number;
    name: string;
    email: string;
}

interface ButtonProps {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary'; // Optional prop
}

// Component với Props và TypeScript
const Button: React.FC<ButtonProps> = ({ label, onClick, variant = 'primary' }) => {
    const style = {
        backgroundColor: variant === 'primary' ? '#667eea' : '#6c757d',
        color: 'white',
        padding: '12px 24px',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
    };

    return (
        <button style={style} onClick={onClick}>
            {label}
        </button>
    );
};

// Component với State
const Counter: React.FC = () => {
    const [count, setCount] = useState<number>(0);

    return (
        <div>
            <p>Count: {count}</p>
            <Button label="Tăng" onClick={() => setCount(count + 1)} />
            <Button label="Giảm" onClick={() => setCount(count - 1)} variant="secondary" />
        </div>
    );
};

// Component với useEffect (Fetch Data)
const UserList: React.FC = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        fetch('https://jsonplaceholder.typicode.com/users')
            .then(res => res.json())
            .then(data => {
                setUsers(data);
                setLoading(false);
            });
    }, []); // [] = chỉ chạy 1 lần khi mount

    if (loading) return <p>Loading...</p>;

    return (
        <ul>
            {users.map(user => (
                <li key={user.id}>{user.name} - {user.email}</li>
            ))}
        </ul>
    );
};

// --- 3. BÀI TẬP (EXERCISE) ---

/**
 * BÀI 1: Tạo component Card nhận props: title, description, imageUrl
 * Hiển thị một card với hình ảnh, tiêu đề và mô tả
 */
interface CardProps {
    // Định nghĩa props tại đây
}

const Card: React.FC<CardProps> = (props) => {
    // Viết code tại đây
    return <div>TODO: Implement Card</div>;
};

/**
 * BÀI 2: Tạo component TodoList với:
 * - Input để nhập todo mới
 * - Button để thêm todo
 * - Danh sách hiển thị các todo
 * - Có thể xóa todo khi click
 */
interface Todo {
    id: number;
    text: string;
}

const TodoList: React.FC = () => {
    const [todos, setTodos] = useState<Todo[]>([]);
    const [input, setInput] = useState<string>('');

    // Viết các hàm addTodo, deleteTodo tại đây

    return (
        <div>
            {/* Viết JSX tại đây */}
            <p>TODO: Implement TodoList</p>
        </div>
    );
};

// Export để sử dụng
export { Button, Counter, UserList, Card, TodoList };
