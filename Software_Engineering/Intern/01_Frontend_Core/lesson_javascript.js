/**
 * ================================================================
 * SE INTERN - LESSON 2: JAVASCRIPT FUNDAMENTALS
 * ================================================================
 */

// --- 1. THEORY ---
/**
 * 1. Variables:
 *    - const: Cannot be reassigned (preferred)
 *    - let: Can be reassigned
 *    - var: Function-scoped (avoid)
 * 
 * 2. Data Types:
 *    - Primitive: string, number, boolean, null, undefined, symbol
 *    - Reference: object, array, function
 * 
 * 3. Functions:
 *    - Regular function: function name() {}
 *    - Arrow function: const name = () => {}
 *    - Arrow functions don't have their own 'this'
 * 
 * 4. Async/Await:
 *    - Promise: Represents a future value
 *    - async function: Returns a Promise
 *    - await: Waits for Promise to resolve
 * 
 * 5. ES6+ Features:
 *    - Destructuring, Spread operator
 *    - Template literals, Optional chaining
 *    - Modules (import/export)
 */

// --- 2. CODE SAMPLE ---

// ========== VARIABLES & DATA TYPES ==========
const name = "John";
let age = 25;
const isStudent = true;
const hobbies = ["coding", "reading", "gaming"];
const person = { name: "John", age: 25 };

// ========== DESTRUCTURING ==========
const { name: personName, age: personAge } = person;
const [firstHobby, ...restHobbies] = hobbies;

console.log(personName); // "John"
console.log(firstHobby); // "coding"
console.log(restHobbies); // ["reading", "gaming"]

// ========== FUNCTIONS ==========
// Regular function
function greet(name) {
    return `Hello, ${name}!`;
}

// Arrow function
const greetArrow = (name) => `Hello, ${name}!`;

// Function with default parameter
const greetWithDefault = (name = "Guest") => `Hello, ${name}!`;

// ========== ARRAY METHODS ==========
const numbers = [1, 2, 3, 4, 5];

// map: Transform each element
const doubled = numbers.map(n => n * 2); // [2, 4, 6, 8, 10]

// filter: Keep elements that pass condition
const evens = numbers.filter(n => n % 2 === 0); // [2, 4]

// reduce: Accumulate to single value
const sum = numbers.reduce((acc, n) => acc + n, 0); // 15

// find: Get first matching element
const firstEven = numbers.find(n => n % 2 === 0); // 2

// some/every: Check conditions
const hasEven = numbers.some(n => n % 2 === 0); // true
const allPositive = numbers.every(n => n > 0); // true

// ========== ASYNC/AWAIT ==========
// Promise example
const fetchData = () => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve({ id: 1, name: "Data" });
        }, 1000);
    });
};

// Using async/await
const getData = async () => {
    try {
        const data = await fetchData();
        console.log("Data:", data);
        return data;
    } catch (error) {
        console.error("Error:", error);
    }
};

// Fetch API example
const fetchUsers = async () => {
    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/users');
        if (!response.ok) throw new Error('Network error');
        const users = await response.json();
        return users;
    } catch (error) {
        console.error("Failed to fetch:", error);
        return [];
    }
};

// ========== DOM MANIPULATION ==========
// Select elements
// const button = document.querySelector('#submit-btn');
// const inputs = document.querySelectorAll('.input-field');

// Add event listener
// button.addEventListener('click', (event) => {
//     event.preventDefault();
//     console.log('Button clicked!');
// });

// Create and append elements
// const newDiv = document.createElement('div');
// newDiv.textContent = 'New content';
// newDiv.classList.add('new-class');
// document.body.appendChild(newDiv);

// ========== CLASSES (OOP) ==========
class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
    }

    greet() {
        return `Hello, I'm ${this.name}`;
    }

    static createGuest() {
        return new User("Guest", "guest@example.com");
    }
}

class Admin extends User {
    constructor(name, email, role) {
        super(name, email);
        this.role = role;
    }

    greet() {
        return `${super.greet()} and I'm an ${this.role}`;
    }
}

// ========== MODULES ==========
// export const API_URL = 'https://api.example.com';
// export function fetchAPI(endpoint) { ... }
// export default class ApiClient { ... }

// import { API_URL, fetchAPI } from './api.js';
// import ApiClient from './api.js';

// --- 3. EXERCISES ---
/**
 * EXERCISE 1: Array manipulation
 *            - Given array of products with {name, price, category}
 *            - Filter products with price > 100
 *            - Map to get product names only
 *            - Calculate total price with reduce
 * 
 * EXERCISE 2: Async data fetching
 *            - Fetch data from JSONPlaceholder API
 *            - Handle loading and error states
 *            - Display results in console
 * 
 * EXERCISE 3: Todo List
 *            - Create functions: addTodo, removeTodo, toggleComplete
 *            - Use array methods (no mutation)
 *            - Store data in localStorage
 * 
 * EXERCISE 4: Event handling
 *            - Create a form with validation
 *            - Show error messages for invalid inputs
 *            - Submit only when all fields are valid
 */

// --- 4. EXERCISE SOLUTIONS ---

// EXERCISE 1: Array manipulation
const products = [
    { name: "Laptop", price: 1200, category: "Electronics" },
    { name: "Phone", price: 800, category: "Electronics" },
    { name: "Mouse", price: 25, category: "Electronics" },
    { name: "Keyboard", price: 150, category: "Electronics" },
    { name: "Book", price: 20, category: "Education" },
    { name: "Monitor", price: 300, category: "Electronics" }
];

// 1. Filter products with price > 100
const expensiveProducts = products.filter(product => product.price > 100);
console.log("Expensive Products (price > 100):", expensiveProducts);

// 2. Map to get product names only
const productNames = products.map(product => product.name);
console.log("All Product Names:", productNames);

// 3. Calculate total price with reduce
const totalPrice = products.reduce((total, product) => total + product.price, 0);
console.log("Total Price of all products:", totalPrice);

// --- TEST CODE ---
// Test code

console.log("=== JavaScript Fundamentals ===");
console.log(greet("World"));
console.log("Doubled:", doubled);
console.log("Sum:", sum);

const admin = new Admin("Alice", "alice@test.com", "Admin");
console.log(admin.greet());
