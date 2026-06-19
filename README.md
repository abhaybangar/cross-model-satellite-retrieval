# PS11 MERN Optical Image Search

This project is a MERN-style proof of concept for uploading optical images and searching a SAR gallery using DINOv2 embeddings.

## Setup

1. Install backend dependencies:
   - Open `c:\Users\banga\Desktop\ps_11_proto\backend`
   - Run `npm install`

2. Install frontend dependencies:
   - Open `c:\Users\banga\Desktop\ps_11_proto\frontend`
   - Run `npm install`

3. Prepare Python dependencies in your Python environment:
   - `pip install pillow torch transformers numpy`

4. Start backend:
   - `cd backend`
   - `node server.js`

5. Start frontend:
   - `cd frontend`
   - `npm start`

## Usage

- Upload an optical image in the frontend.
- The backend runs `backend/python/search.py` and returns the top matches from `dataset/`.
- Matching images are displayed in the frontend.

## Notes

- The backend serves images from `dataset/`.
- If you want MongoDB logging, set `MONGO_URI` before starting the backend.
