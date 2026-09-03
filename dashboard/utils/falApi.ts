/**
 * Simple helper to make authenticated requests to FAL API
 * Routes through Next.js API proxy which adds Authorization: Key $FAL_KEY header
 */

/**
 * Make an authenticated request to FAL API through proxy
 */
export async function authenticatedFetch(targetUrl: string, options: RequestInit = {}) {
  if (!targetUrl.includes('fal.run')) {
    return fetch(targetUrl, {
      method: options.method || 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: (options.method || 'POST') === 'GET' ? undefined : (options.body || JSON.stringify({})),
    })
  }

  return fetch('/api/fal/proxy', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Fal-Target-Url': targetUrl,
      'X-Fal-Method': options.method || 'POST',
    },
    body: options.body || JSON.stringify({}), // Default to empty object if no body provided
  });
}

/**
 * Helper functions for common API calls
 */
export async function startStream(apiUrl: string, config: any) {
  return authenticatedFetch(`${apiUrl}/start_stream`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function stopStream(apiUrl: string) {
  return authenticatedFetch(`${apiUrl}/stop_stream`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function updateConfig(apiUrl: string, config: any) {
  return authenticatedFetch(`${apiUrl}/update_config`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function getMetrics(apiUrl: string) {
  return authenticatedFetch(`${apiUrl}/metrics`, {
    method: 'GET',
    // No body for GET requests
  });
}

