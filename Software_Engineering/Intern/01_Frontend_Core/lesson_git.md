# SE INTERN - BÀI 4: GIT & GITHUB

## --- 1. LÝ THUYẾT (THEORY) ---

### Git là gì?
- **Version Control System (VCS)**: Quản lý lịch sử thay đổi của code
- **Repository (Repo)**: Thư mục chứa project và lịch sử Git
- **Commit**: Một "snapshot" của code tại thời điểm cụ thể
- **Branch**: Nhánh phát triển độc lập

### Workflow cơ bản
```
Working Directory → Staging Area → Local Repo → Remote Repo
     (edit)           (add)         (commit)       (push)
```

---

## --- 2. CODE MẪU (COMMANDS) ---

### Khởi tạo và Clone
```bash
# Khởi tạo repo mới
git init

# Clone repo từ GitHub
git clone https://github.com/username/repo.git
```

### Commit Flow
```bash
# Xem trạng thái file
git status

# Thêm file vào staging
git add filename.js      # Thêm 1 file
git add .                # Thêm tất cả

# Commit với message
git commit -m "feat: add login feature"

# Xem lịch sử commit
git log --oneline
```

### Branch & Merge
```bash
# Tạo và chuyển sang branch mới
git checkout -b feature/login

# Liệt kê branches
git branch

# Chuyển branch
git checkout main

# Merge branch vào main
git merge feature/login
```

### Push & Pull
```bash
# Đẩy code lên remote
git push origin main

# Kéo code mới về
git pull origin main
```

### Undo Changes
```bash
# Bỏ file khỏi staging
git reset HEAD filename.js

# Hoàn tác thay đổi (chưa commit)
git checkout -- filename.js

# Sửa commit message gần nhất
git commit --amend -m "new message"
```

---

## --- 3. BÀI TẬP (EXERCISE) ---

### BÀI 1: Thực hành cơ bản
1. Tạo một thư mục mới và `git init`
2. Tạo file `README.md` và commit với message "Initial commit"
3. Thêm file `index.html` và commit với message "Add homepage"
4. Xem lịch sử commit bằng `git log --oneline`

### BÀI 2: Branching
1. Tạo branch `feature/about-page`
2. Thêm file `about.html` và commit
3. Quay lại `main` và tạo branch `feature/contact-page`
4. Thêm file `contact.html` và commit
5. Merge cả 2 branch vào `main`

### BÀI 3: Giải quyết Conflict
1. Tạo 2 branch khác nhau
2. Sửa cùng 1 dòng trong cùng 1 file ở cả 2 branch
3. Merge và giải quyết conflict

---

## Commit Message Convention

```
<type>: <subject>

Types:
- feat: Tính năng mới
- fix: Sửa bug
- docs: Thay đổi documentation
- style: Format code (không ảnh hưởng logic)
- refactor: Tái cấu trúc code
- test: Thêm/sửa tests
- chore: Các thay đổi khác (build, config)

Ví dụ:
feat: add user authentication
fix: resolve login redirect issue
docs: update API documentation
```
