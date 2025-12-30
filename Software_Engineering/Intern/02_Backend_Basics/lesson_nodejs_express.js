/**
 * ================================================================
 * SE INTERN - LESSON 5: NODE.JS + EXPRESS.JS
 * ================================================================
 * 
 * Install: npm init -y && npm install express
 * Run: node lesson_nodejs_express.js
 */

const express = require('express');
const app = express();

// --- 1. THEORY ---
/**
 * 1. Node.js:
 *    - JavaScript runtime outside browser
 *    - Event-driven, non-blocking I/O
 *    - npm: Package manager
 * 
 * 2. Express.js:
 *    - Popular Node.js web framework
 *    - Middleware-based architecture
 *    - Routing and HTTP methods
 * 
 * 3. Middleware:
 *    - Functions that run between request and response
 *    - Order matters! Runs top to bottom
 *    - Types: Application, Router, Error-handling
 * 
 * 4. HTTP Methods:
 *    - GET: Read data
 *    - POST: Create data
 *    - PUT: Update entire resource
 *    - PATCH: Partial update
 *    - DELETE: Remove data
 */

// --- 2. CODE SAMPLE ---

// ========== MIDDLEWARE ==========

// Built-in middleware
app.use(express.json());  // Parse JSON body
app.use(express.urlencoded({ extended: true }));  // Parse form data

// Custom logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();  // Pass to next middleware
});

// ========== IN-MEMORY DATABASE ==========
let products = [
    { id: 1, name: 'Laptop', price: 999, stock: 10 },
    { id: 2, name: 'Phone', price: 699, stock: 25 },
    { id: 3, name: 'Tablet', price: 499, stock: 15 },
];

let nextId = 4;

// ========== ROUTES ==========

// GET all products
app.get('/api/products', (req, res) => {
    // Query params: ?minPrice=500&maxPrice=1000
    const { minPrice, maxPrice, search } = req.query;

    let result = [...products];

    if (minPrice) result = result.filter(p => p.price >= Number(minPrice));
    if (maxPrice) result = result.filter(p => p.price <= Number(maxPrice));
    if (search) result = result.filter(p =>
        p.name.toLowerCase().includes(search.toLowerCase())
    );

    res.json({
        success: true,
        count: result.length,
        data: result
    });
});

// GET single product
app.get('/api/products/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const product = products.find(p => p.id === id);

    if (!product) {
        return res.status(404).json({
            success: false,
            error: 'Product not found'
        });
    }

    res.json({ success: true, data: product });
});

// POST create product
app.post('/api/products', (req, res) => {
    const { name, price, stock } = req.body;

    // Validation
    if (!name || !price) {
        return res.status(400).json({
            success: false,
            error: 'Name and price are required'
        });
    }

    const newProduct = {
        id: nextId++,
        name,
        price: Number(price),
        stock: Number(stock) || 0
    };

    products.push(newProduct);

    res.status(201).json({
        success: true,
        data: newProduct
    });
});

// PUT update product
app.put('/api/products/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const index = products.findIndex(p => p.id === id);

    if (index === -1) {
        return res.status(404).json({
            success: false,
            error: 'Product not found'
        });
    }

    const { name, price, stock } = req.body;
    products[index] = { id, name, price: Number(price), stock: Number(stock) };

    res.json({ success: true, data: products[index] });
});

// DELETE product
app.delete('/api/products/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const index = products.findIndex(p => p.id === id);

    if (index === -1) {
        return res.status(404).json({
            success: false,
            error: 'Product not found'
        });
    }

    const deleted = products.splice(index, 1)[0];

    res.json({
        success: true,
        message: `Product "${deleted.name}" deleted`
    });
});

// ========== ERROR HANDLING ==========
app.use((err, req, res, next) => {
    console.error('Error:', err.message);
    res.status(500).json({
        success: false,
        error: 'Internal server error'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: `Route ${req.method} ${req.path} not found`
    });
});

// ========== START SERVER ==========
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

// --- 3. EXERCISES ---
/**
 * EXERCISE 1: Add validation middleware
 *            - Validate required fields
 *            - Validate data types
 *            - Return 400 for invalid data
 * 
 * EXERCISE 2: Add pagination
 *            - Query params: ?page=1&limit=10
 *            - Return meta: { page, limit, total, totalPages }
 * 
 * EXERCISE 3: Add Router
 *            - Separate routes into /routes/products.js
 *            - Use express.Router()
 * 
 * EXERCISE 4: Add authentication middleware
 *            - Check for Authorization header
 *            - Validate token format
 *            - Allow only authenticated users to POST/PUT/DELETE
 */
