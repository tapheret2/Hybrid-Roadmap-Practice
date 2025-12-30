/**
 * ================================================================
 * SE INTERN - BÀI 2: JAVASCRIPT FUNDAMENTALS
 * ================================================================
 */

// --- 1. LÝ THUYẾT (THEORY) ---
/**
 * 1. Variables: let (block-scoped), const (immutable), var (function-scoped, tránh dùng)
 * 2. Data Types: string, number, boolean, null, undefined, object, array
 * 3. Functions: Regular function, Arrow function (=>)
 * 4. DOM Manipulation: document.querySelector(), addEventListener()
 * 5. Async: Promise, async/await, fetch API
 * 6. ES6+: Destructuring, Spread operator, Template literals
 */

// --- 2. CODE MẪU (CODE SAMPLE) ---

// Variables & Types
const appName = "My App";
let count = 0;
const users = ["An", "Bình", "Chi"];

// Arrow Function
const greet = (name) => `Xin chào, ${name}!`;
console.log(greet("Bạn")); // Output: Xin chào, Bạn!

// Array Methods (rất quan trọng!)
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);        // [2, 4, 6, 8, 10]
const evens = numbers.filter(n => n % 2 === 0); // [2, 4]
const sum = numbers.reduce((acc, n) => acc + n, 0); // 15

// Destructuring
const person = { name: "An", age: 22 };
const { name, age } = person;

// Spread Operator
const newNumbers = [...numbers, 6, 7]; // [1, 2, 3, 4, 5, 6, 7]

// Async/Await với Fetch
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Lỗi:", error);
    }
}

// DOM Manipulation (chạy trong browser)
// document.querySelector(".btn").addEventListener("click", () => {
//     alert("Button clicked!");
// });

// --- 3. BÀI TẬP (EXERCISE) ---

/**
 * BÀI 1: Viết hàm tính tổng các số chẵn trong mảng
 * Input: [1, 2, 3, 4, 5, 6]
 * Output: 12 (2 + 4 + 6)
 */
function sumEvenNumbers(arr) {
    // Gợi ý: Dùng filter và reduce
}

/**
 * BÀI 2: Viết hàm fetch dữ liệu từ API và trả về tên của user đầu tiên
 * API: https://jsonplaceholder.typicode.com/users
 */
async function getFirstUserName() {
    // Gợi ý: Dùng fetch, await, và destructuring
}

/**
 * BÀI 3: Tạo một object chứa thông tin sản phẩm (name, price, quantity)
 * Viết hàm tính tổng giá trị đơn hàng (price * quantity)
 */
function calculateOrderTotal(product) {
    // Viết code tại đây
}

// --- TEST ---
console.log("=== Chạy file này bằng Node.js: node lesson_javascript.js ===");
// console.log(sumEvenNumbers([1, 2, 3, 4, 5, 6])); // Expected: 12
