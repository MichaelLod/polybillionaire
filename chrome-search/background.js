// PolyBillionaire Search Proxy v1.2
//
// Uses real Chrome tabs to search Google (fully rendered, no CORS, no CAPTCHA).
// The service worker opens a tab, waits for it to load, injects a content
// script to extract results, then closes the tab.

const BRIDGE_URL = "http://localhost:3456";

// Open a URL in a new tab, wait for load, execute script, close tab
function executeInTab(url, scriptFunc, timeout = 20000) {
  return new Promise((resolve, reject) => {
    let tabId = null;
    let timer = null;

    timer = setTimeout(() => {
      if (tabId) chrome.tabs.remove(tabId).catch(() => {});
      reject(new Error("Tab execution timeout"));
    }, timeout);

    chrome.tabs.create({ url, active: false }, (tab) => {
      tabId = tab.id;

      // Wait for the page to finish loading
      function onUpdated(updatedTabId, changeInfo) {
        if (updatedTabId !== tabId || changeInfo.status !== "complete") return;
        chrome.tabs.onUpdated.removeListener(onUpdated);

        // Small delay to let JS render
        setTimeout(() => {
          chrome.scripting.executeScript(
            { target: { tabId }, func: scriptFunc },
            (results) => {
              clearTimeout(timer);
              chrome.tabs.remove(tabId).catch(() => {});
              if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
              } else {
                resolve(results?.[0]?.result || null);
              }
            }
          );
        }, 1500); // 1.5s for JS to render
      }

      chrome.tabs.onUpdated.addListener(onUpdated);
    });
  });
}

// Content script: extract Google search results from rendered DOM
function extractGoogleResults() {
  const results = [];
  const maxResults = 8;

  // Method 1: div.g blocks (standard Google results)
  document.querySelectorAll("div.g").forEach((block) => {
    if (results.length >= maxResults) return;
    const link = block.querySelector("a[href]");
    const title = block.querySelector("h3");
    if (!link || !title) return;

    const url = link.href;
    if (!url || url.includes("google.com/search")) return;

    // Find snippet text
    let snippet = "";
    const spans = block.querySelectorAll("span");
    for (const span of spans) {
      const text = span.textContent.trim();
      if (text.length > 40 && text !== title.textContent.trim()) {
        snippet = text;
        break;
      }
    }

    results.push({
      title: title.textContent.trim(),
      url: url,
      snippet: snippet.substring(0, 200),
    });
  });

  // Method 2: Fallback — any <a> with <h3> child
  if (results.length === 0) {
    document.querySelectorAll("a").forEach((a) => {
      if (results.length >= maxResults) return;
      const h3 = a.querySelector("h3");
      if (!h3) return;
      const url = a.href;
      if (!url || url.includes("google.com")) return;
      results.push({
        title: h3.textContent.trim(),
        url: url,
        snippet: "",
      });
    });
  }

  return results;
}

// Content script: extract text content from any page
function extractPageText() {
  // Remove noise
  document
    .querySelectorAll("script, style, nav, footer, header, aside, iframe")
    .forEach((el) => el.remove());

  let text = document.body ? document.body.innerText : "";
  return text.substring(0, 8000);
}

// Execute a Google search via real tab
async function executeSearch(query, maxResults = 5) {
  try {
    const url = `https://www.google.com/search?q=${encodeURIComponent(query)}&num=${maxResults}&hl=en&gl=us`;
    const results = await executeInTab(url, extractGoogleResults);
    console.log(`[PB] Search "${query}" → ${(results || []).length} results`);
    return results || [];
  } catch (e) {
    console.error(`[PB] Search error: ${e.message}`);
    return [{ title: `Search error: ${e.message}`, url: "", snippet: "" }];
  }
}

// Fetch page text via real tab
async function executeFetch(url) {
  try {
    const text = await executeInTab(url, extractPageText);
    console.log(`[PB] Fetch "${url}" → ${(text || "").length} chars`);
    return text || "[empty page]";
  } catch (e) {
    console.error(`[PB] Fetch error: ${e.message}`);
    return `[fetch error: ${e.message}]`;
  }
}

// Serial execution queue — only one tab operation at a time
let busy = false;

// Poll bridge for pending requests
async function pollBridge() {
  if (busy) return;  // Skip if already handling a request
  try {
    const resp = await fetch(`${BRIDGE_URL}/pending`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    if (resp.status === 204) return;
    if (!resp.ok) return;

    const request = await resp.json();
    const { id, type, query, url, max_results } = request;
    console.log(`[PB] Got request: ${type} — ${query || url}`);
    busy = true;

    let result;
    try {
      if (type === "search") {
        result = await executeSearch(query, max_results || 5);
      } else if (type === "fetch") {
        result = await executeFetch(url);
      } else {
        result = { error: `Unknown type: ${type}` };
      }
    } finally {
      busy = false;
    }

    await fetch(`${BRIDGE_URL}/results`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, result }),
    });
    console.log(`[PB] Completed ${id}`);
  } catch (e) {
    // Bridge not running — silent
  }
}

// Keep-alive: chrome.alarms fires every 25s to prevent service worker suspension.
// The fast setTimeout loop handles responsive polling while alive.
chrome.alarms.create("pb-keepalive", { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "pb-keepalive") {
    pollBridge();
  }
});

// Fast poll loop — setTimeout chain for <1s response time
let pollActive = true;

function pollLoop() {
  if (!pollActive) return;
  pollBridge().finally(() => {
    setTimeout(pollLoop, 1000);
  });
}

// Restart fast loop on service worker wake
pollLoop();
console.log("[PB] PolyBillionaire Search Proxy v1.3 started (tab-based, keep-alive)");
