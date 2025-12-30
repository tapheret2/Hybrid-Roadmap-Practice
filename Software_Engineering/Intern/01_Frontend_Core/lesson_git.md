# SE INTERN - LESSON 4: GIT & GITHUB

## 1. THEORY

### What is Git?
- **Version Control System**: Track changes in code over time
- **Distributed**: Every developer has a complete history
- **Branching**: Create separate lines of development

### Key Concepts
- **Repository (Repo)**: Project folder with .git history
- **Commit**: A snapshot of changes
- **Branch**: Independent line of development
- **Merge**: Combine branches together
- **Remote**: Server copy (GitHub, GitLab)

---

## 2. CODE SAMPLE (Commands)

### Initial Setup
```bash
# Configure user info (one time)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Initialize new repository
git init

# Clone existing repository
git clone https://github.com/user/repo.git
```

### Basic Workflow
```bash
# Check current status
git status

# Stage changes
git add filename.js       # Single file
git add .                 # All changes

# Commit changes
git commit -m "feat: add login functionality"

# Push to remote
git push origin main
```

### Branching
```bash
# Create and switch to new branch
git checkout -b feature/login

# Switch between branches
git checkout main
git checkout feature/login

# Merge branch into main
git checkout main
git merge feature/login

# Delete branch after merge
git branch -d feature/login
```

### Syncing with Remote
```bash
# Fetch latest changes (no merge)
git fetch origin

# Pull (fetch + merge)
git pull origin main

# Push changes
git push origin feature/login
```

### Handling Conflicts
```bash
# When merge conflict occurs:
1. Open conflicting files
2. Look for conflict markers: <<<<<<< ======= >>>>>>>
3. Edit to keep desired code
4. Remove conflict markers
5. git add .
6. git commit -m "fix: resolve merge conflict"
```

---

## Commit Message Convention

### Format
```
<type>: <description>

[optional body]
[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructure (no behavior change)
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```
feat: add user authentication
fix: resolve login redirect issue
docs: update API documentation
refactor: simplify order validation logic
```

---

## 3. EXERCISES

### Exercise 1: Basic Git Workflow
1. Create a new repository
2. Add README.md file
3. Make 3 commits with proper messages
4. Push to GitHub

### Exercise 2: Branching
1. Create branch `feature/header`
2. Add header component
3. Create branch `feature/footer`
4. Add footer component
5. Merge both into main

### Exercise 3: Collaboration
1. Fork a public repository
2. Clone to local machine
3. Create a feature branch
4. Make changes and push
5. Create a Pull Request

### Exercise 4: Conflict Resolution
1. Create two branches that modify the same file
2. Merge first branch into main
3. Try to merge second branch (will conflict)
4. Resolve the conflict manually
5. Complete the merge

---

## Useful Commands Reference

```bash
# View commit history
git log --oneline -n 10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git checkout -- .

# Stash changes temporarily
git stash
git stash pop

# View differences
git diff
git diff --staged

# Rebase (alternative to merge)
git rebase main
```
