# Home Monitor — Setup Guide

## What you need
- Render account (free tier works)
- Neon account with a project (free tier works)
- GitHub account + repo
- Anthropic API key (claude.ai → API → Keys)
- 2 spare Android/iPhone phones with a browser

---

## Step 1 — Push to GitHub

```bash
cd C:\Users\liamm\home-monitor
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/home-monitor.git
git push -u origin main
```

---

## Step 2 — Get your Neon connection string

1. Go to [console.neon.tech](https://console.neon.tech)
2. Open your project → **Connection Details**
3. Copy the **Connection string** (starts with `postgresql://...`)

---

## Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) → **New** → **Blueprint**
2. Connect your GitHub repo — Render will detect `render.yaml` automatically
3. In the service settings, add two **Environment Variables**:

   | Key | Value |
   |-----|-------|
   | `DATABASE_URL` | Your Neon connection string |
   | `ANTHROPIC_API_KEY` | Your Anthropic API key |

4. Click **Apply** — Render will build and deploy (takes ~2 min)
5. Your app URL will be: `https://home-monitor.onrender.com`

---

## Step 4 — Set up the mobile cameras

On each phone:

1. Open the browser (Chrome on Android, Safari on iPhone)
2. Navigate to: `https://home-monitor.onrender.com/camera`
3. **Add to Home Screen** so it works like an app:
   - Android: tap the 3-dot menu → *Add to Home screen*
   - iPhone: tap Share → *Add to Home Screen*
4. Open the camera app, enter the room name (e.g. `Kitchen`), tap **Start Monitoring**
5. Allow camera access when prompted

Each phone will automatically capture a frame and send it for AI analysis on the interval you choose (default: every 20 seconds).

---

## Step 5 — Enrol family members

1. Open `https://home-monitor.onrender.com` on any device
2. Under **Enrol a Person**: type the person's name, upload a clear photo of their face
3. Click **Add Person** — Claude AI will analyse the face and store a description
4. Repeat for each family member

---

## How it works

```
Phone camera → captures frame every 20s
     ↓
POST /api/frame  (image + location name)
     ↓
Claude Vision API:
  • Matches face against enrolled profiles
  • Describes what the person is doing
     ↓
Neon PostgreSQL:
  • Saves sighting (who, task, location, timestamp, thumbnail)
     ↓
Dashboard auto-refreshes every 15s to show activity
```

---

## Cost estimate (free tiers)

| Service | Free allowance | Expected usage |
|---------|---------------|---------------|
| Render | 750 hrs/month | Fits on free |
| Neon | 0.5 GB storage | Fine for months |
| Anthropic | Pay-per-use | ~$0.003 per frame (Sonnet) |

At 20-second intervals on 2 phones: ~8,640 frames/day × $0.003 ≈ **~$26/day**.  
**Tip:** Set interval to 60 s to reduce to ~$9/day, or only run cameras when needed.

---

## Troubleshooting

**Camera not starting on iPhone**: Make sure you're on HTTPS (Render provides this). Safari requires HTTPS for camera access.

**"Unknown" person on every frame**: The enrolled face description may not be detailed enough. Try re-enrolling with a clearer, well-lit front-facing photo.

**Render service sleeping (free tier)**: Render free web services sleep after 15 min of inactivity. The first camera frame after sleep will take ~30 s to respond. Upgrade to Render Starter ($7/mo) to avoid this.
