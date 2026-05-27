// CyberGuard AI Service Worker v2
const CACHE_VERSION = 'cyberguard-v2';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const API_CACHE = `${CACHE_VERSION}-api`;

const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
    '/offline.html',
];

const MAX_DYNAMIC_CACHE = 100;

// Install — pre-cache static assets
self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => {
            console.log('[SW] Pre-caching static assets');
            return cache.addAll(STATIC_ASSETS);
        })
    );
});

// Activate — clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys
                    .filter((k) => k.startsWith('cyberguard-') && !k.startsWith(CACHE_VERSION))
                    .map((k) => {
                        console.log('[SW] Removing old cache:', k);
                        return caches.delete(k);
                    })
            )
        ).then(() => self.clients.claim())
    );
});

function isStaticAsset(pathname) {
    return /\.(js|css|woff2?|ttf|eot|svg|png|jpg|jpeg|gif|webp|ico)$/i.test(pathname);
}

async function cacheFirst(request, cacheName) {
    const cached = await caches.match(request);
    if (cached) return cached;
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, response.clone());
        }
        return response;
    } catch (_e) {
        return new Response('Offline', { status: 503 });
    }
}

async function networkFirst(request, cacheName) {
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, response.clone());
            trimCache(cacheName, MAX_DYNAMIC_CACHE);
        }
        return response;
    } catch (_e) {
        const cached = await caches.match(request);
        return cached || new Response('Offline', { status: 503 });
    }
}

async function staleWhileRevalidate(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    const fetchPromise = fetch(request).then((response) => {
        if (response.ok) cache.put(request, response.clone());
        return response;
    }).catch(() => cached);
    return cached || fetchPromise;
}

async function networkFirstWithFallback(request) {
    try {
        const response = await fetch(request);
        const cache = await caches.open(DYNAMIC_CACHE);
        cache.put(request, response.clone());
        return response;
    } catch (_e) {
        const cached = await caches.match(request);
        if (cached) return cached;
        return caches.match('/offline.html');
    }
}

async function trimCache(cacheName, maxItems) {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    if (keys.length > maxItems) {
        await cache.delete(keys[0]);
        trimCache(cacheName, maxItems);
    }
}

// Fetch strategies
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    if (request.method !== 'GET') return;

    if (url.pathname.startsWith('/api/')) {
        event.respondWith(staleWhileRevalidate(request, API_CACHE));
        return;
    }
    if (isStaticAsset(url.pathname)) {
        event.respondWith(cacheFirst(request, STATIC_CACHE));
        return;
    }
    if (request.mode === 'navigate') {
        event.respondWith(networkFirstWithFallback(request));
        return;
    }
    event.respondWith(networkFirst(request, DYNAMIC_CACHE));
});

// Push notifications
self.addEventListener('push', (event) => {
    let data = { title: 'CyberGuard AI Alert', body: 'Yeni bir güvenlik uyarısı var.' };
    try { data = event.data.json(); } catch (_e) { data.body = event.data?.text() || data.body; }

    event.waitUntil(
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: '/icons/icon-192x192.png',
            badge: '/icons/icon-72x72.png',
            vibrate: [100, 50, 100],
            tag: data.tag || 'cyberguard-alert',
            data: { url: data.url || '/' },
            actions: [
                { action: 'view', title: 'Görüntüle' },
                { action: 'dismiss', title: 'Kapat' },
            ],
        })
    );
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    if (event.action === 'dismiss') return;
    const url = event.notification.data?.url || '/';
    event.waitUntil(
        self.clients.matchAll({ type: 'window' }).then((clients) => {
            for (const client of clients) {
                if (client.url.includes(url) && 'focus' in client) return client.focus();
            }
            return self.clients.openWindow(url);
        })
    );
});

// Background sync
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-pending-reports') {
        event.waitUntil(syncPendingData());
    }
});

async function syncPendingData() {
    try {
        const cache = await caches.open('cyberguard-pending');
        const requests = await cache.keys();
        for (const request of requests) {
            const response = await cache.match(request);
            const body = await response.text();
            await fetch(request, { method: 'POST', body, headers: { 'Content-Type': 'application/json' } });
            await cache.delete(request);
        }
    } catch (_e) {
        console.log('[SW] Sync failed, will retry');
    }
}
