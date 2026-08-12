import { useRef, useState } from "react";
import { askQuestion } from "../api";
import Message from "./Message.jsx";

export default function ChatBox() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi, I'm Medibot. Ask a medical question about the loaded document." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const outputRef = useRef(null);

  async function send(query) {
    if (!query.trim() || busy) return;
    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setBusy(true);

    try {
      const data = await askQuestion(query);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: data.answer,
          correctedQuery: data.corrected_query,
          citations: data.citations,
          followUps: data.follow_up_questions,
          metrics: data.metrics,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => {
        if (outputRef.current) {
          outputRef.current.scrollTop = outputRef.current.scrollHeight;
        }
      });
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    send(input);
  }

  return (
    <div className="chat-box-wrapper">
      <div className="chat-box" ref={outputRef}>
        {messages.map((m, i) => (
          <Message key={i} message={m} onFollowUpClick={send} />
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a medical question about the loaded document."
        />
        <button type="submit" disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
