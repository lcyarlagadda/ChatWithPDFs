# PDF Chat Frontend - Setup Guide

## Prerequisites

Before you begin, make sure you have:
- **Node.js** (version 18 or higher) - [Download here](https://nodejs.org/)
- **npm** (comes with Node.js) or **yarn**
- A running backend server (see backend setup instructions)

## Quick Start

### Step 1: Install Node.js

If you don't have Node.js installed:

1. Go to https://nodejs.org/
2. Download the LTS (Long Term Support) version
3. Install it (use default settings)
4. Verify installation:
```bash
node --version
npm --version
```

### Step 2: Install Dependencies

Open a terminal in the `frontend` directory and run:

```bash
npm install
```

This will install all required packages (React, Next.js, Tailwind CSS, etc.)

### Step 3: Configure Backend URL

Create a `.env.local` file in the `frontend` directory:

**Option A: Using a text editor**
1. Create a new file named `.env.local` in the `frontend` folder
2. Add this line (replace with your actual backend URL):
```
NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io
```

**Option B: Using command line**

Windows (PowerShell):
```powershell
echo "NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io" > .env.local
```

Mac/Linux:
```bash
echo "NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io" > .env.local
```

**Important:** Replace `https://your-ngrok-url.ngrok.io` with the actual URL from your Colab backend!

### Step 4: Run the Development Server

```bash
npm run dev
```

You should see output like:
```
- ready started server on 0.0.0.0:3000, url: http://localhost:3000
- event compiled client and server successfully
```

### Step 5: Open in Browser

Open your browser and go to:
```
http://localhost:3000
```

You should see the PDF Chat System interface!

## Common Issues & Solutions

### Issue 1: "Command not found: npm"
**Solution:** Node.js is not installed or not in your PATH
- Reinstall Node.js
- Restart your terminal/command prompt
- Check with: `node --version`

### Issue 2: "Port 3000 is already in use"
**Solution:** Something else is using port 3000
```bash
# Run on a different port
npm run dev -- -p 3001
```
Then open: `http://localhost:3001`

### Issue 3: "Module not found" errors
**Solution:** Dependencies not installed properly
```bash
# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json
# Reinstall
npm install
```

### Issue 4: Backend connection fails
**Solution:** Check your backend URL
1. Make sure your Colab backend is running
2. Check the ngrok URL is correct in `.env.local`
3. Make sure the URL includes `https://` (not `http://`)
4. Restart the dev server after changing `.env.local`

### Issue 5: "TypeError: Failed to fetch"
**Solution:** Backend URL is incorrect or backend is not running
- Verify backend is running in Colab
- Check the ngrok URL is still active (free ngrok URLs expire)
- Make sure there are no typos in the URL

## File Structure

```
frontend/
├── app/                      # Next.js app directory
│   ├── components/          # React components
│   │   ├── ChatInterface.tsx
│   │   ├── FileUpload.tsx
│   │   └── SettingsPanel.tsx
│   ├── services/           # API services
│   │   └── pdfChatService.ts
│   ├── globals.css         # Global styles
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Main page
├── public/                  # Static files
├── .env.local              # Environment variables (create this)
├── package.json            # Dependencies
├── tailwind.config.js      # Tailwind CSS config
├── next.config.js          # Next.js config
├── tsconfig.json           # TypeScript config
└── README.md               # This file
```

## Available Scripts

### `npm run dev`
Runs the app in development mode
- Hot reload enabled (changes appear instantly)
- Open http://localhost:3000

### `npm run build`
Builds the app for production
- Creates optimized production build
- Output in `.next` folder

### `npm run start`
Runs the production build
- Must run `npm run build` first
- Use this for production deployment

### `npm run lint`
Runs ESLint to check code quality
- Finds potential bugs
- Enforces code style

## Environment Variables

Create `.env.local` in the frontend directory:

```env
# Backend API URL (from your Colab ngrok or deployed backend)
NEXT_PUBLIC_API_URL=https://your-backend-url.ngrok.io

# Optional: For production deployment
# NEXT_PUBLIC_API_URL=https://your-production-backend.com
```

**Note:** 
- Prefix with `NEXT_PUBLIC_` to expose to browser
- Restart dev server after changing `.env.local`
- Never commit `.env.local` to git (it's in .gitignore)

## Step-by-Step Tutorial

### Complete Setup from Scratch

1. **Open Terminal/Command Prompt**
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - Mac: Press `Cmd + Space`, type "Terminal", press Enter
   - Or use VS Code's integrated terminal

2. **Navigate to Frontend Folder**
   ```bash
   cd path/to/ChatWithPDFs/frontend
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```
   Wait for installation to complete (may take 2-5 minutes)

4. **Create Environment File**
   Create a file named `.env.local` with your backend URL:
   ```
   NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io
   ```

5. **Start Development Server**
   ```bash
   npm run dev
   ```

6. **Open Browser**
   Go to `http://localhost:3000`

7. **Start Using**
   - Upload PDF files
   - Wait for processing
   - Ask questions!

## Deployment to Production

### Option 1: Vercel (Recommended)

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your repository
5. Set environment variable:
   - Name: `NEXT_PUBLIC_API_URL`
   - Value: Your backend URL
6. Click Deploy

### Option 2: Netlify

1. Push code to GitHub
2. Go to [netlify.com](https://netlify.com)
3. Click "Add new site" → "Import existing project"
4. Select your repository
5. Build settings:
   - Build command: `npm run build`
   - Publish directory: `.next`
6. Add environment variable: `NEXT_PUBLIC_API_URL`
7. Deploy

### Option 3: Manual Build

```bash
# Build for production
npm run build

# Run production server
npm run start
```

## Troubleshooting Checklist

- [ ] Node.js is installed (check with `node --version`)
- [ ] You're in the `frontend` directory
- [ ] Dependencies are installed (`npm install` was successful)
- [ ] `.env.local` file exists with correct backend URL
- [ ] Backend is running (check Colab notebook)
- [ ] ngrok URL is correct and active
- [ ] No firewall blocking the connection
- [ ] Browser console shows no CORS errors (F12 to open)

## Getting Help

If you're still having issues:

1. **Check the browser console** (F12 → Console tab)
   - Look for error messages
   - Share them when asking for help

2. **Check the terminal output**
   - Look for compilation errors
   - Note any warning messages

3. **Verify backend is working**
   - Open backend URL in browser: `https://your-ngrok-url.ngrok.io/health`
   - Should show: `{"status":"healthy","timestamp":"..."}`

4. **Common fixes**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules .next
   npm install
   npm run dev
   ```

## Next Steps

Once the app is running:

1. **Upload PDFs**: Click the upload area and select PDF files
2. **Wait for Processing**: Processing may take 1-2 minutes first time (models loading)
3. **Ask Questions**: Type questions about your documents
4. **Adjust Settings**: Click the settings icon to customize behavior
5. **Export Conversations**: Use the export button to save chat history

## Tips for Best Experience

- Use Chrome, Firefox, or Edge for best compatibility
- Keep the Colab session active while using the app
- For large PDFs, be patient during processing
- Try different settings to optimize responses
- Use specific questions for better answers

## Need More Help?

- Check `DEPLOYMENT.md` for detailed deployment instructions
- Review the main `README.md` for architecture overview
- Open an issue on GitHub if you find bugs

