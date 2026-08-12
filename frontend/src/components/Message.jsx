export default function Message({ message, onFollowUpClick }) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${isUser ? "user" : "bot"}`}>
      <div className="message-content">
        <p>{message.text}</p>

        {message.correctedQuery && (
          <p className="spelling-hint">
            Did you mean: <em>{message.correctedQuery}</em>?
          </p>
        )}

        {message.citations?.length > 0 && (
          <div className="citations">
            <strong>Sources:</strong>
            <ul>
              {message.citations.map((c, i) => (
                <li key={i}>
                  {c.filename} — page {c.page}
                </li>
              ))}
            </ul>
          </div>
        )}

        {message.followUps?.length > 0 && (
          <div className="follow-ups">
            {message.followUps.map((q, i) => (
              <button key={i} className="follow-up-chip" onClick={() => onFollowUpClick(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        {message.metrics && (
          <p className="metrics">
            {message.metrics.total_ms.toFixed(0)}ms · {message.metrics.chunks_used} chunk(s)
          </p>
        )}
      </div>
    </div>
  );
}
