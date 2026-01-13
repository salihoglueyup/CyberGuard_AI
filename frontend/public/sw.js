// CyberGuard AI Service Worker
const CACHE_NAME = 'cyberguard-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/assets/index.css',
    '/manifest.json'
];

// Install event
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('CyberGuard: Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
});

// Fetch event - Network first, fallback to cache
self.addEventListener('fetch', (event) => {
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // Clone the response
                const responseClone = response.clone();

                caches.open(CACHE_NAME)
                    .then((cache) => {
                        // Only cache GET requests
                        if (event.request.method === 'GET') {
                            cache.put(event.request, responseClone);
                        }
                    });

                return response;
            })
            .catch(() => {
                // If network fails, try cache
                return caches.match(event.request);
            })
    );
});

// Activate event - Clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('CyberGuard: Deleting old cache', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Push notification support
self.addEventListener('push', (event) => {
    const options = {
        body: event.data.text(),
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            { action: 'explore', title: 'View Dashboard', icon: '/icons/dashboard.png' },
            { action: 'close', title: 'Close', icon: '/icons/close.png' }
        ]
    };

    event.waitUntil(
        self.registration.showNotification('CyberGuard AI Alert', options)
    );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'explore') {
        clients.openWindow('/');
    }
});
