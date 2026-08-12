const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function askQuestion(query) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Query failed");
  }
  return res.json();
}
