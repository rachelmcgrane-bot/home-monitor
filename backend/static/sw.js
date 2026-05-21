const CACHE = 'home-monitor-v3';
const PRECACHE = [
  '/static/manifest.json',
];

// Install: pre-cache only the manifest (avoid caching pages on install — they may be mid-deploy)
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(PRECACHE).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

// Activate: delete all old caches immediately
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch strategy:
//  - API calls      → network only (never cache live data)
//  - HTML pages (/, /camera, /nfc*) → network only (always fresh from server)
//  - Static assets  → cache first, fall back to network
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Always network for API and NFC endpoints
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/nfc')) {
    e.respondWith(fetch(e.request).catch(() =>
      new Response(JSON.stringify({error:'offline'}), {headers:{'Content-Type':'application/json'}})
    ));
    return;
  }

  // HTML pages: always fetch fresh (no cache — prevents stale app shell)
  if (e.request.mode === 'navigate' || e.request.headers.get('accept')?.includes('text/html')) {
    e.respondWith(fetch(e.request).catch(() =>
      new Response('<h2 style="font-family:system-ui;padding:2rem;color:#fff;background:#0d0f1a">Offline — please reconnect and refresh.</h2>',
        {headers:{'Content-Type':'text/html'}})
    ));
    return;
  }

  // Static assets (JS, CSS, fonts, images): cache first
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(resp => {
        if (resp && resp.status === 200 && resp.type !== 'opaque') {
          const clone = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return resp;
      }).catch(() => cached || new Response('Offline', {status: 503}));
    })
  );
});
