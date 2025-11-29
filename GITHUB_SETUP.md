# GitHub Setup Instructions

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `sticker-creator` (or any name you prefer)
3. Description: "Create stickers with white outline and automatic background removal"
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **Create repository**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/sticker-creator.git

# Or if you prefer SSH:
# git remote add origin git@github.com:YOUR_USERNAME/sticker-creator.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create sticker-creator --public --source=. --remote=origin --push
```

## Step 3: Verify

Check your repository at: `https://github.com/YOUR_USERNAME/sticker-creator`

## Notes

- The `rembg` library is NOT included in the repository (it's too large - hundreds of MB)
- Users will be prompted to install it automatically on first run
- All dependencies are listed in `requirements.txt`
- Users can run `install_dependencies.bat` to install everything at once

