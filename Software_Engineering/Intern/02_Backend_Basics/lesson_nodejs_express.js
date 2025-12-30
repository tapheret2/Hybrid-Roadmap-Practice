/**
 * ================================================================
 * SE INTERN - BACKEND: NODE.JS + EXPRESS.JS
 * ================================================================
 * 
 * Cài đặt: npm init -y && npm install express
 * Chạy: node lesson_nodejs_express.js
 */

// --- 1. LÝ THUYẾT (THEORY) ---
/**
 * 1. Express.js: Web framework cho Node.js, xử lý HTTP requests
 * 2. Middleware: Hàm xử lý request trước khi đến route handler
 * 3. Routing: Định nghĩa cách ứng dụng phản hồi với các endpoint
 * 4. Request/Response: req chứa data từ client, res gửi data về client
 * 5. HTTP Methods: GET (đọc), POST (tạo), PUT (cập nhật), DELETE (xóa)
 */

const express = require('express');
const app = express();
const PORT = 3000;

// --- 2. CODE MẪU (CODE SAMPLE) ---

// Middleware: Parse JSON body
app.use(express.json());

// Middleware: Logging (tự viết)
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next(); // Chuyển sang middleware/route tiếp theo
});

// In-memory database (giả lập)
let users = [
    { id: 1, name: 'An', email: 'an@example.com' },
    { id: 2, name: 'Bình', email: 'binh@example.com' },
];

// GET - Lấy tất cả users
app.get('/api/users', (req, res) => {
    res.json(users);
});

// GET - Lấy user theo ID
app.get('/api/users/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const user = users.find(u => u.id === id);

    if (!user) {
        return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
});

// POST - Tạo user mới
app.post('/api/users', (req, res) => {
    const { name, email } = req.body;

    if (!name || !email) {
        return res.status(400).json({ error: 'Name and email are required' });
    }

    const newUser = {
        id: users.length + 1,
        name,
        email
    };
    users.push(newUser);
    res.status(201).json(newUser);
});

// PUT - Cập nhật user
app.put('/api/users/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const userIndex = users.findIndex(u => u.id === id);

    if (userIndex === -1) {
        return res.status(404).json({ error: 'User not found' });
    }

    users[userIndex] = { ...users[userIndex], ...req.body };
    res.json(users[userIndex]);
});

// DELETE - Xóa user
app.delete('/api/users/:id', (req, res) => {
    const id = parseInt(req.params.id);
    users = users.filter(u => u.id !== id);
    res.status(204).send();
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});

// --- 3. BÀI TẬP (EXERCISE) ---
/**
 * BÀI 1: Thêm route GET /api/users/search?name=xxx để tìm user theo tên
 * 
 * BÀI 2: Tạo một middleware xác thực API key từ header x-api-key
 *        Nếu không có hoặc sai key → trả về 401 Unauthorized
 * 
 * BÀI 3: Tạo CRUD cho resource "products" với các fields:
 *        - id, name, price, category
 *        - Các routes: GET /api/products, POST /api/products, etc.
 */
